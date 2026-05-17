import logging
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import re
from ragas import evaluate
from ragas.metrics import ResponseRelevancy, LLMContextPrecisionWithoutReference, Faithfulness, AnswerAccuracy, ContextRelevance, ResponseGroundedness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import EvaluationDataset

# A simple regex to find JSON objects in a string
JSON_FINDER_REGEX = r"```json\n({.*?})\n```"

class RAGEvaluator:
    """
    A class to evaluate the performance of a Retrieval-Augmented Generation (RAG) system.
    It uses a large language model to perform qualitative assessments.
    """
    def __init__(self, llm, embeddings):
        """
        Initializes the evaluator with a language model.
        Args:
            llm: The language model to use for evaluations.
        """
        self.llm = llm
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()
        self.embeddings = embeddings
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _run_chain(self, prompt_template: str, input_data: Dict[str, Any], parser):
        """Helper to run a evaluation chain."""
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | parser
            return chain.invoke(input_data)
        except Exception as e:
            self.logger.error(f"Error running evaluation chain: {e}")
            # Fallback for parsing errors
            if parser == self.json_parser:
                try:
                    # Try to extract JSON manually from the raw output
                    raw_output = (prompt | self.llm | self.str_parser).invoke(input_data)
                    json_match = re.search(JSON_FINDER_REGEX, raw_output, re.DOTALL)
                    if json_match:
                        return self.json_parser.parse(json_match.group(1))
                except Exception as inner_e:
                    self.logger.error(f"Fallback parsing failed: {inner_e}")
            return None

    def evaluate_faithfulness(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluates if the answer is factually consistent with the provided context.
        """
        prompt = """
        You are an expert evaluator. Your task is to assess the faithfulness of a generated answer to the provided context.
        The answer is faithful if all claims made in it are supported by the context.
        Score the faithfulness on a scale from 0.0 to 1.0, where 1.0 is completely faithful and 0.0 means it contains hallucinations.
        Provide a brief justification for your score, noting any specific claims that are not supported by the context.

        Question: {question}
        Context: {context}
        Answer: {answer}

        Return your response as a JSON object with "score" and "justification" keys.
        Example: ```json
        {{
            "score": 0.7,
            "justification": "The answer correctly identifies the main pathways but mentions a treatment not present in the context."
        }}
        ```
        """
        return self._run_chain(prompt, {"question": question, "answer": answer, "context": context}, self.json_parser)

    def evaluate_context_precision(self, question: str, context: str) -> Dict[str, Any]:
        """
        Evaluates if the retrieved context is relevant and concise for answering the question.
        """
        prompt = """
        You are an expert evaluator. Your task is to assess the precision of the retrieved context in relation to the user's question.
        Precision measures the signal-to-noise ratio. A high precision score means most of the information in the context is useful for answering the question.
        Score the precision on a scale from 0.0 to 1.0.
        Provide a brief justification for your score.

        Question: {question}
        Context: {context}

        Return your response as a JSON object with "score" and "justification" keys.
        Example: ```json
        {{
            "score": 0.8,
            "justification": "The context contains highly relevant documents, but also includes some tangential information about clinical trial administration."
        }}
        ```
        """
        return self._run_chain(prompt, {"question": question, "context": context}, self.json_parser)

    def evaluate_context_recall(self, question: str, context: str, ground_truth_answer: str) -> Dict[str, Any]:
        """
        Evaluates if the context contains all the information needed to answer the question, based on a ground truth answer.
        """
        if not ground_truth_answer:
            return {"score": "N/A", "justification": "Ground truth answer not provided."}

        prompt = """
        You are an expert evaluator. Your task is to assess the recall of the retrieved context.
        Recall measures if all the necessary information to construct the 'Ground Truth Answer' was present in the context.
        Score the recall on a scale from 0.0 to 1.0.
        Provide a brief justification, noting any key pieces of information from the ground truth answer that are missing from the context.

        Question: {question}
        Ground Truth Answer: {ground_truth_answer}
        Context: {context}

        Return your response as a JSON object with "score" and "justification" keys.
        Example: ```json
        {{
            "score": 0.6,
            "justification": "The context covers the main genetic mutations but is missing information about their impact on specific treatment options mentioned in the ground truth."
        }}
        ```
        """
        return self._run_chain(prompt, {"question": question, "context": context, "ground_truth_answer": ground_truth_answer}, self.json_parser)

    def evaluate_contextual_graph_coherence(self, nodedetails: Dict[str, Any], entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        NOVEL METRIC: Evaluates the coherence of the retrieved graph context.
        A coherent graph has relevant entities that are well-connected.
        """
        try:
            # Extract entity and relationship IDs
            entity_ids = set(entities.get("entityids", []))
            relationship_ids = set(entities.get("relationshipids", []))

            num_nodes = len(entity_ids)
            num_relationships = len(relationship_ids)

            if num_nodes < 2:
                return {
                    "score": 0.0,
                    "justification": f"Graph is trivial with {num_nodes} node(s) and {num_relationships} relationship(s).",
                    "details": {"nodes": num_nodes, "relationships": num_relationships}
                }

            # Calculate graph density: 2 * E / (V * (V - 1)) for an undirected graph
            # This measures how many of the possible connections between nodes actually exist.
            max_possible_edges = num_nodes * (num_nodes - 1)
            density = (2 * num_relationships) / max_possible_edges if max_possible_edges > 0 else 0
            
            # The score is the density, capped at 1.0
            score = min(density, 1.0)

            justification = (
                f"The retrieved subgraph shows moderate coherence. "
                if score > 0.5 else
                f"The retrieved subgraph shows low coherence. "
            )
            justification += f"Calculated a density score of {score:.2f} based on {num_nodes} entities and {num_relationships} relationships."

            return {
                "score": round(score, 4),
                "justification": justification,
                "details": {"nodes": num_nodes, "relationships": num_relationships, "density": round(density, 4)}
            }
        except Exception as e:
            self.logger.error(f"Error calculating graph coherence: {e}")
            return {"score": 0.0, "justification": f"An error occurred: {e}", "details": {}}

    def evaluate_using_ragas(self, metric_details: Dict[str, Any], ground_truth_answer: str = None) -> Dict[str, Any]:
        """
        Evaluates answer relevancy using the ragas library.
        """
        try:
            question = metric_details.get("question")
            context = metric_details.get("contexts")
            answer = metric_details.get("answer")
            nodedetails = metric_details.get("nodedetails", {})
            entities = metric_details.get("entities", {})

            if not all([question, context, answer]):
                return {"error": "Missing one or more required fields: question, contexts, answer."}

            self.logger.info("Starting RAG evaluation...")

            evaluator_llm = LangchainLLMWrapper(self.llm)
            evaluator_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

            if not all([question, answer, context]):
                raise ValueError("Question, answer, and context must be non-empty.")

            # Prepare data in required format
            data_sample = {
                'user_input': question,
                'retrieved_contexts': [context],  # Nested list because each question can have multiple contexts
                'response': answer,
            }
            dataset = EvaluationDataset.from_list([data_sample])

            # Evaluate using ragas
            result = evaluate(
                dataset=dataset,
                metrics=[ResponseRelevancy(), Faithfulness(), LLMContextPrecisionWithoutReference(), ContextRelevance(), ResponseGroundedness()],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            )

            # Optional: Log score
            self.logger.info(f"Ragas response relevancy score: {result}")
            self.logger.info("RAG evaluation completed.")

            print(type(result))
            return result

        except Exception as e:
            self.logger.error(f"Error during Ragas answer relevancy evaluation: {e}")
            return {"score": 0.0, "justification": f"An error occurred: {e}"}

    def evaluate_all(self, metric_details: Dict[str, Any], ground_truth_answer: str = None) -> Dict[str, Any]:
        """
        Runs all evaluation metrics.
        """
        question = metric_details.get("question")
        context = metric_details.get("contexts")
        answer = metric_details.get("answer")
        nodedetails = metric_details.get("nodedetails", {})
        entities = metric_details.get("entities", {})

        if not all([question, context, answer]):
            return {"error": "Missing one or more required fields: question, contexts, answer."}

        self.logger.info("Starting RAG evaluation...")

        results = {
            "answer_relevancy": self.evaluate_answer_relevancy_ragas(question, answer, context),
            "faithfulness": self.evaluate_faithfulness(question, answer, context),
            "context_precision": self.evaluate_context_precision(question, context),
            "context_recall": self.evaluate_context_recall(question, context, ground_truth_answer),
            "contextual_graph_coherence": self.evaluate_contextual_graph_coherence(nodedetails, entities)
        }

        self.logger.info("RAG evaluation completed.")
        return results
