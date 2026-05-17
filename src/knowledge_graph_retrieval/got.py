import logging
import time
from typing import Dict, Any, List
import re
from openai import OpenAI
import csv

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.QA_integration import (
    get_llm, get_neo4j_retriever, create_document_retriever_chain,
    retrieve_documents, format_documents, get_rag_chain, EMBEDDING_FUNCTION
)
from src.metrics.evaluation import RAGEvaluator
from src.constants import FORMAT_ANSWER_PROMPT

# Prompts for Graph of Thoughts
THOUGHT_GENERATION_PROMPT = """
You are an expert at reasoning and breaking down complex questions.
Given the original question and the current state of gathered evidence, generate a set of diverse and insightful next-step questions or hypotheses to explore.
These "thoughts" should aim to uncover new information, connect existing evidence, or validate hypotheses.

Original Question: {question}
Current Evidence:
{evidence}

Generate 4-6 distinct thoughts. Each thought should be a concise question or a statement to investigate.
Return your response as a JSON object with the following structure:
{{
    "thoughts": [
        "thought 1",
        "thought 2",
        ...
    ]
}}
"""

THOUGHT_EVALUATION_PROMPT = """
You are a critical thinker and evaluator.
Given the original question and a set of generated "thoughts", evaluate the potential of each thought to contribute to a comprehensive answer.
Assign a score from 0.0 to 1.0 to each thought, where 1.0 is the most promising.

Original Question: {question}
Generated Thoughts:
{thoughts}

Return your response as a JSON object with the following structure:
{{
    "scores": [
        {{"thought": "thought 1", "score": 0.9}},
        {{"thought": "thought 2", "score": 0.7}},
        ...
    ]
}}
"""

MHOP_SYNTHESIS_PROMPT = """
You are an AI assistant that processes a question along with a given context
Your job is to extract the final answer only in the shortest possible form.

Rules:
1. Always read the full context before answering.
2. Identify the user’s question.
3. Use only the provided context — never use external knowledge.
5. Based on the context, respond to the user’s question with:
   - One word (e.g., "Yes", "No", "Phoolwari", "Iran")
   - Or the shortest possible phrase if a single word is not sufficient.
6. Do not include anything besides the exact final answer

Original Question: {original_question}

Context:
{evidence_summary}

Answer:
"""

SYNTHESIS_PROMPT = """
You are a master synthesizer of information.
Given the original question and a collection of evidence gathered from a graph of thoughts, synthesize a comprehensive and coherent final answer.
Structure your answer logically, addressing all aspects of the original question.

Original Question: {original_question}

Evidence from Thought Exploration:
{evidence_summary}

Synthesized Answer:
"""

SYNTHESIS_PROMPT_ultradomain = """
You are a precision synthesis expert who transforms validated evidence into comprehensive, actionable answers that excel in both depth and practical utility. Your responses must be accurate, thorough, and strictly grounded in the provided context.

⚠️ **Critical Constraint**: You may ONLY use information from the provided context. Never use external knowledge or generate speculative answers.

### WINNING RESPONSE FORMULA:

**1. Lead with Concrete Detail (Critical for Success)**
- Start with specific examples, numbers, and data points from evidence
- Include step-by-step procedures or methodologies when available
- Name specific techniques, tools, people, or places mentioned
- Provide exact measurements, percentages, or quantitative data
- Use real-world applications and case studies from the evidence

**2. Comprehensive Depth Before Gaps**
- Cover the topic exhaustively using available evidence FIRST
- Explain mechanisms, processes, and cause-effect relationships
- Include historical context and development when present
- Provide thorough practical implementation details
- Only mention limitations after substantial content (80/20 rule)

**3. Actionable Empowerment**
- Transform concepts into "how-to" guidance wherever possible
- Include specific next steps readers can take
- Explain practical applications alongside theoretical concepts
- Provide concrete tools, resources, or methods from evidence
- Frame information for immediate implementation

**4. Diverse Perspectives (Integrated, Not Dominant)**
- Weave multiple viewpoints naturally throughout the response
- Include varied evidence types without making gaps the focus
- Present different angles while maintaining practical focus
- Balance critical analysis with substantial content

**5. Strategic Structure**
- Lead with most actionable/concrete information
- Build from specific examples to broader principles
- Use clear headings for easy navigation
- Progressive disclosure: essential → detailed → nuanced
- Conclude with practical synthesis, not just gaps

### SYNTHESIS APPROACH:

1. **First Pass - Concrete Foundation**
   - Extract ALL specific data, examples, numbers
   - Identify step-by-step processes or procedures
   - List concrete tools, methods, or applications

2. **Second Pass - Comprehensive Build**
   - Expand each concrete element with full context
   - Connect related evidence for complete picture
   - Add mechanisms and explanations

3. **Third Pass - Practical Integration**
   - Transform knowledge into actionable guidance
   - Highlight implementation pathways
   - Ensure every section has practical value

4. **Final Pass - Strategic Polish**
   - Verify 80% content / 20% limitations ratio
   - Ensure concrete examples appear early
   - Check that gaps enhance rather than dominate
   - Confirm actionable takeaways throughout

### RESPONSE CHECKLIST:
Before finalizing, ensure you have:
- [ ] Specific numbers, names, or concrete examples in first paragraph
- [ ] Step-by-step guidance or clear procedures
- [ ] Exhaustive coverage of what IS known
- [ ] Practical applications clearly explained
- [ ] Multiple perspectives woven throughout (not separate)
- [ ] Gaps/limitations as minor concluding notes only
- [ ] Clear actionable takeaways for readers

### Question:
{original_question}

### Context:
<context>
{evidence_summary}
</context>

Remember: Win through DEPTH + DETAIL + ACTIONABILITY. Comprehensive answers with concrete specifics and practical guidance consistently outperform analytical frameworks alone. Your goal is to be the most useful possible resource while maintaining analytical rigor.
"""


class GraphOfThoughtsRetriever:
    def __init__(self, llm, graph, document_names, chat_mode_settings):
        self.llm = llm
        self.graph = graph
        self.document_names = document_names
        self.chat_mode_settings = chat_mode_settings
        self.retriever = get_neo4j_retriever(
            graph=graph,
            chat_mode_settings=chat_mode_settings,
            document_names=document_names
        )
        self.doc_retriever = create_document_retriever_chain(
            llm, self.retriever)

    def generate_thoughts(self, question: str, evidence: str) -> List[str]:
        prompt = ChatPromptTemplate.from_template(
            THOUGHT_GENERATION_PROMPT)
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({"question": question, "evidence": evidence})
            return result.get("thoughts", [])
        except Exception as e:
            logging.error(f"Error generating thoughts: {e}")
            return [question]

    def evaluate_thoughts(self, question: str, thoughts: List[str]) -> List[Dict[str, Any]]:
        prompt = ChatPromptTemplate.from_template(
            THOUGHT_EVALUATION_PROMPT)
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({"question": question, "thoughts": thoughts})
            return sorted(result.get("scores", []), key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logging.error(f"Error evaluating thoughts: {e}")
            return [{"thought": thought, "score": 0.5} for thought in thoughts]

    def retrieve_for_thought(self, thought: str) -> Dict[str, Any]:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=thought)]
        docs, _ = retrieve_documents(self.doc_retriever, messages)

        if docs:
            formatted_docs, sources, entities, communities = format_documents(
                docs, self.llm.model_name, self.chat_mode_settings
            )
            rag_chain = get_rag_chain(self.llm)
            ai_response = rag_chain.invoke({
                "messages": [],
                "context": formatted_docs,
                "input": thought
            })
            return {
                "thought": thought,
                "answer": ai_response.content,
                "sources": list(sources),
                "entities": entities,
                "communities": communities,
                "formatted_docs": formatted_docs
            }
        else:
            return {
                "thought": thought,
                "answer": "No relevant information found for this thought.",
                "sources": [],
                "entities": {},
                "communities": [],
                "formatted_docs": ""
            }

    def synthesize_answer_deepseek(self, question: str, evidence_summary: str) -> str:
        prompt = SYNTHESIS_PROMPT_ultradomain.format(
            original_question=question, evidence_summary=evidence_summary)

        response = deepseek_client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a Question Answering assistant that uses the context provided and answers the question asked.  "},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    def synthesize_answer_gpt(self, question: str, evidence_summary: str) -> str:
        prompt = SYNTHESIS_PROMPT_ultradomain.format(
            original_question=question, evidence_summary=evidence_summary)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Question Answering assistant that uses the context provided and answers the question asked.  "},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    def synthesize_answer(self, question: str, evidence_summary: str):
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT_ultradomain)
        chain = prompt | self.llm

        formatted_prompt = SYNTHESIS_PROMPT_ultradomain.format(
            original_question=question, evidence_summary=evidence_summary)
        response = chain.invoke({
            "original_question": question,
            "evidence_summary": evidence_summary
        })

        return response.content, formatted_prompt

    def format_answer(self, question: str, detailed_answer: str):
        prompt = ChatPromptTemplate.from_template(FORMAT_ANSWER_PROMPT)
        chain = prompt | self.llm

        response = chain.invoke({
            "question": question,
            "detailed_answer": detailed_answer
        })

        return response.content

    def remove_entities_and_relationships(self, text):
        # First, let's preserve the document structure by processing each document separately
        documents = text.split('Document start')
        processed_documents = []

        for doc in documents:
            if not doc.strip():
                continue

            # Remove only the Entities section (more precise pattern)
            doc = re.sub(
                r'\n----\s*\nEntities:\s*\n(?:Entity:[^\n]*\n)*', '\n', doc, flags=re.MULTILINE)

            # Remove only the Relationships section (more precise pattern)
            doc = re.sub(
                r'\n----\s*\nRelationships:\s*\n(?:Entity:[^\n]*\n)*', '\n', doc, flags=re.MULTILINE)

            # # Alternative: If entities/relationships might have different formats, use this more flexible approach
            # # This looks for the section headers and removes until the next section marker or document end
            # doc = re.sub(r'\n----\s*\nEntities:.*?(?=\n----|\nDocument end|$)', '', doc, flags=re.DOTALL)
            # doc = re.sub(r'\n----\s*\nRelationships:.*?(?=\nDocument end|$)', '', doc, flags=re.DOTALL)

            processed_documents.append(doc)

        # Rejoin documents
        result = 'Document start'.join(processed_documents)

        # Clean up any leftover separators and extra whitespace
        result = re.sub(r'----\s*\n(?=Document end)', '', result)
        result = re.sub(r'\n{3,}', '\n\n', result)

        # Ensure we don't have trailing/leading whitespace issues
        result = result.strip()

        return result

    def conduct_graph_of_thought(
        self,
        question: str,
        max_depth: int = 3  # 2 → total 6 thoughts, 3 → total 9 thoughts
    ):
        all_retrieved_info = []

        # Step 1: Generate and evaluate initial 3 thoughts
        initial_thoughts = self.generate_thoughts(
            question, "Initial question.")
        evaluated_thoughts = self.evaluate_thoughts(question, initial_thoughts)

        # Keep exactly 3 initial thoughts
        queue = [thought['thought'] for thought in evaluated_thoughts[:3]]

        # Step 2: Expand each thought only once per depth
        for depth in range(1, max_depth + 1):
            logging.info(f"Depth {depth}: Expanding {len(queue)} thoughts")

            next_queue = []

            # Process each thought exactly once
            for current_thought in queue:
                with open("output.txt", "a") as file:
                    file.write(f"Current Thought {current_thought} \n")

                logging.info(f"Processing thought: {current_thought}")

                retrieved_info = self.retrieve_for_thought(current_thought)
                all_retrieved_info.append(retrieved_info)

                # Build evidence so far
                current_evidence = "\n\n".join([
                    f"Thought: {info['thought']}\nContent: {info['formatted_docs']}"
                    for info in all_retrieved_info
                ])
                formatted_current_evidence = self.remove_entities_and_relationships(
                    current_evidence)

                # Generate only one new thought for this branch
                new_thoughts = self.generate_thoughts(
                    question, formatted_current_evidence)
                evaluated_new_thoughts = self.evaluate_thoughts(
                    question, new_thoughts)

                # Pick only the top thought per parent
                if evaluated_new_thoughts:
                    next_queue.append(evaluated_new_thoughts[0]['thought'])

            # Prepare for next depth
            queue = next_queue

        # Step 3: Summarize evidence
        evidence_summary = "\n---\n".join(
            [
                f"Thought: {info['thought']}\nAnswer: {info['answer']}\nContent: {info['formatted_docs']}"
                for info in all_retrieved_info
            ]
        )
        formatted_evidence_summary = self.remove_entities_and_relationships(
            evidence_summary)

        # Step 4: Synthesize final answer
        final_answer = self.synthesize_answer_gpt(
            question, formatted_evidence_summary)

        return final_answer, all_retrieved_info


def graph_of_thought_qa(graph, model, question, document_names, chat_mode_settings):
    start_time = time.time()

    try:
        llm, model_name = get_llm(model=model)
        retriever = GraphOfThoughtsRetriever(
            llm=llm,
            graph=graph,
            document_names=document_names,
            chat_mode_settings=chat_mode_settings
        )

        final_answer, all_retrieved_info = retriever.conduct_graph_of_thought(
            question)

        all_sources = set()
        all_entities = {"entityids": set(), "relationshipids": set()}
        all_formatted_docs = []

        for info in all_retrieved_info:
            all_sources.update(info["sources"])
            if "entityids" in info.get("entities", {}):
                all_entities["entityids"].update(info["entities"]["entityids"])
            if "relationshipids" in info.get("entities", {}):
                all_entities["relationshipids"].update(
                    info["entities"]["relationshipids"])
            all_formatted_docs.append(info["formatted_docs"])

        all_formatted_docs = "\n---\n".join(all_formatted_docs)

        metrics = {}

        if chat_mode_settings.get("evaluation", False):
            # --- Run RAGAS-based Evaluation ---
            evaluator = RAGEvaluator(llm=llm, embeddings=EMBEDDING_FUNCTION)
            metric_details = {
                "question": question,
                "contexts": all_formatted_docs,
                "answer": final_answer,
                "nodedetails": all_entities.get("nodedetails", {}),
                "entities": all_entities.get("entities", {})
            }
            # For a real scenario, you might have a ground truth dataset for context_recall
            ground_truth_answer = None
            metrics = evaluator.evaluate_using_ragas(
                metric_details, ground_truth_answer)
            # --------------------

        total_time = time.time() - start_time

        return {
            "message": final_answer,
            "info": {
                "metrics": metrics,
                "sources": list(all_sources),
                "model": model_name,
                "mode": "graph_of_thought",
                "response_time": total_time,
                "thought_process": all_retrieved_info,
                "entities": {
                    "entityids": list(all_entities["entityids"]),
                    "relationshipids": list(all_entities["relationshipids"])
                },
            },
            "user": "chatbot"
        }

    except Exception as e:
        logging.exception(f"Error in Graph of Thought QA: {str(e)}")
        return {
            "message": "An error occurred during the Graph of Thought retrieval process.",
            "info": {
                "error": str(e),
                "mode": "graph_of_thought"
            },
            "user": "chatbot"
        }
