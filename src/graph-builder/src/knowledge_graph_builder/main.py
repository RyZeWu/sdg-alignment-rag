"""
Knowledge graph builder for SDG 6 policy-alignment documents.
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add the parent directory to the Python path for imports.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.knowledge_graph_builder.config import load_config
from src.knowledge_graph_builder.llm import call_configured_llm, extract_json_from_text
from src.knowledge_graph_builder.text_utils import chunk_text
from src.knowledge_graph_builder.entity_standardization import (
    standardize_entities,
    infer_relationships,
    limit_predicate_length,
)
from src.knowledge_graph_builder.prompts import (
    SDG_POLICY_SYSTEM_PROMPT,
    SDG_POLICY_USER_PROMPT,
)
from src.knowledge_graph_builder.embedding import get_embedding, load_embedding_model
from src.knowledge_graph_builder.utils.pdf_chunking import pdf_to_text

SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_PDF_SUFFIXES = {".pdf"}


def _safe_document_id(source_path, input_root):
    """Create a stable file-system-safe document id from the input path."""
    try:
        relative_path = source_path.relative_to(input_root)
    except ValueError:
        relative_path = Path(source_path.name)

    stem_parts = relative_path.with_suffix("").parts
    raw_id = "__".join(stem_parts)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id).strip("_")
    return safe_id or source_path.stem


def _path_is_under(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _iter_supported_inputs(input_path, output_dir):
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    supported_suffixes = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_PDF_SUFFIXES

    if input_path.is_file():
        if input_path.suffix.lower() in supported_suffixes:
            return [input_path]
        return []

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = []
    for candidate in sorted(input_path.rglob("*")):
        if not candidate.is_file():
            continue
        if _path_is_under(candidate.resolve(), output_dir):
            continue
        if candidate.suffix.lower() in supported_suffixes:
            files.append(candidate)
    return files


def _prepare_input_documents(config, input_path, output_dir, experiment_name):
    """
    Return document records for text files and PDFs.

    PDFs are converted once into an experiment text-input folder. Existing text files
    are used directly. Each record keeps the original source path for provenance.
    """
    input_root = Path(input_path).expanduser().resolve()
    if input_root.is_file():
        input_root = input_root.parent

    output_dir = Path(output_dir).expanduser().resolve()
    text_input_dir = output_dir / f"{experiment_name}_text_inputs"
    convert_pdfs = config.get("documents", {}).get("convert_pdfs", True)
    documents = []

    for source_path in _iter_supported_inputs(input_path, output_dir):
        suffix = source_path.suffix.lower()
        document_id = _safe_document_id(source_path, input_root)

        if suffix in SUPPORTED_PDF_SUFFIXES:
            if not convert_pdfs:
                print(f"Skipping PDF because PDF conversion is disabled: {source_path}")
                continue

            text_input_dir.mkdir(parents=True, exist_ok=True)
            text_path = text_input_dir / f"{document_id}.txt"

            try:
                needs_conversion = not text_path.exists() or (
                    source_path.stat().st_mtime > text_path.stat().st_mtime
                )
                if needs_conversion:
                    print(f"Converting PDF to text: {source_path}")
                    pdf_to_text(str(source_path), str(text_path))
            except Exception as exc:
                print(f"Warning: failed to convert PDF {source_path}: {exc}")
                continue

            source_type = "pdf"
        else:
            text_path = source_path
            source_type = "text"

        documents.append(
            {
                "document_id": document_id,
                "text_path": str(text_path),
                "source_path": str(source_path),
                "source_file": source_path.name,
                "source_type": source_type,
            }
        )

    return documents


def _document_metadata(config, document):
    domain_config = config.get("domain", {})
    return {
        "document_id": document["document_id"],
        "source_file": document["source_file"],
        "source_path": document["source_path"],
        "source_type": document["source_type"],
        "domain": domain_config.get("name", "sdg6_policy_alignment"),
        "sdg_focus": domain_config.get("sdg_focus", "sdg 6"),
    }


def _extraction_budget_instruction(config):
    extraction_config = config.get("extraction", {})
    raw_limit = extraction_config.get("max_triples_per_chunk", 0)
    try:
        max_triples = int(raw_limit or 0)
    except (TypeError, ValueError):
        max_triples = 0

    if max_triples <= 0:
        return ""

    return (
        f"\nExtraction budget:\n"
        f"- Return at most {max_triples} high-value triples for this chunk.\n"
        "- Prioritize SDG 6 targets/indicators, policy instruments, actors, "
        "financing, implementation, monitoring, and measurable outcomes.\n"
        "- Ignore generic background, references, publication metadata, "
        "repeated boilerplate, and weak co-occurrences.\n"
    )


def process_with_llm(config, input_text, debug=False, metadata=None):
    """
    Process input text with the configured LLM to extract triples.

    Args:
        config: Configuration dictionary
        input_text: Text to analyze
        debug: If True, print detailed debug information
        metadata: Optional provenance fields to attach to each triple

    Returns:
        List of extracted triples or None if processing failed
    """
    system_prompt = SDG_POLICY_SYSTEM_PROMPT
    user_prompt = SDG_POLICY_USER_PROMPT
    user_prompt += _extraction_budget_instruction(config)
    user_prompt += f"```\n{input_text}```\n"

    llm_start = time.monotonic()
    response = call_configured_llm(config, user_prompt, system_prompt)
    llm_elapsed = time.monotonic() - llm_start
    print(f"LLM call completed in {llm_elapsed:.1f}s; response length: {len(response.strip())} chars")

    if debug:
        print("Raw LLM response:")
        print(response)
        print("\n---\n")

    if not response.strip():
        print("Error: LLM returned an empty response for this chunk.")
        return None

    result = extract_json_from_text(response)
    if isinstance(result, dict) and isinstance(result.get("triples"), list):
        result = result["triples"]

    if not isinstance(result, list):
        print(
            "\n\nERROR ### Could not extract valid JSON array from response: ",
            response[:1000],
            "\n\n",
        )
        return None

    valid_triples = []
    invalid_count = 0
    metadata = metadata or {}

    for item in result:
        if (
            isinstance(item, dict)
            and "subject" in item
            and "predicate" in item
            and "object" in item
        ):
            triple = dict(item, **metadata)
            triple["predicate"] = limit_predicate_length(triple["predicate"])
            valid_triples.append(triple)
        else:
            invalid_count += 1

    if invalid_count > 0:
        print(
            f"Warning: Filtered out {invalid_count} invalid triples missing required fields"
        )

    if not valid_triples:
        print("Error: No valid triples found in LLM response")
        return None

    if debug:
        print("Extracted JSON:")
        print(json.dumps(valid_triples, indent=2))

    return valid_triples


def _build_hybrid_output(processed_chunks, all_triples, metadata):
    final_output_data = []
    inferred_triples = []
    triples_by_chunk = {chunk["id"]: [] for chunk in processed_chunks}

    for triple in all_triples:
        for key, value in metadata.items():
            triple.setdefault(key, value)
        chunk_id = triple.get("chunk_id")
        if chunk_id in triples_by_chunk:
            triples_by_chunk[chunk_id].append(triple)
        elif triple.get("inferred") is True:
            inferred_triples.append(triple)

    for chunk in processed_chunks:
        chunk_with_triples = dict(chunk)
        chunk_with_triples["triples"] = triples_by_chunk.get(chunk["id"], [])
        final_output_data.append(chunk_with_triples)

    if inferred_triples:
        final_output_data.append(
            {
                "id": f"{metadata.get('document_id', 'document')}-inferred-triples",
                "source": "inferred",
                "triples": inferred_triples,
                **metadata,
            }
        )

    return final_output_data


def _save_hybrid_output(output_file, processed_chunks, all_triples, metadata):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(_build_hybrid_output(processed_chunks, all_triples, metadata), f, indent=2)


def process_text_in_chunks(
    config,
    full_text,
    embedding_model,
    debug=False,
    metadata=None,
):
    """
    Process a large text by chunking it, embedding each chunk, and extracting triples.
    """
    chunk_size = config.get("chunking", {}).get("chunk_size", 700)
    overlap = config.get("chunking", {}).get("overlap", 100)
    text_chunks = chunk_text(full_text, chunk_size, overlap)
    metadata = metadata or {}

    print("=" * 50)
    print("PHASE 1: CHUNK PROCESSING (TRIPLES & EMBEDDINGS)")
    print("=" * 50)
    print(
        f"Processing text in {len(text_chunks)} chunks "
        f"(size: {chunk_size} words, overlap: {overlap} words)"
    )

    all_triples = []
    processed_chunks = []

    for i, chunk in enumerate(text_chunks):
        chunk_id = str(uuid.uuid4())
        chunk_start = time.monotonic()
        print(f"Processing chunk {i + 1}/{len(text_chunks)} ({len(chunk.split())} words)")

        embedding_start = time.monotonic()
        embedding = get_embedding(chunk, embedding_model)
        print(f"Embedding completed in {time.monotonic() - embedding_start:.1f}s")
        if not embedding:
            print(f"Warning: Failed to generate embedding for chunk {chunk_id}. Skipping.")
            continue

        processed_chunks.append(
            {
                "id": chunk_id,
                "chunk_index": i + 1,
                "text": chunk,
                "embedding": embedding,
                **metadata,
            }
        )

        chunk_triples = process_with_llm(config, chunk, debug, metadata)

        if chunk_triples:
            for item in chunk_triples:
                item["chunk_id"] = chunk_id
                item["chunk_index"] = i + 1
            all_triples.extend(chunk_triples)
            print(
                f"Chunk {i + 1}/{len(text_chunks)} produced {len(chunk_triples)} triples "
                f"in {time.monotonic() - chunk_start:.1f}s"
            )
        else:
            print(
                f"Warning: Failed to extract triples from chunk {chunk_id} "
                f"after {time.monotonic() - chunk_start:.1f}s"
            )

    print(f"\nExtracted a total of {len(all_triples)} raw triples from all chunks.")
    return all_triples, processed_chunks


def process_file(config, document, output_file, embedding_model, debug=False):
    """
    Process one prepared document record and write hybrid chunk/triple JSON.
    """
    input_file = document["text_path"]
    metadata = _document_metadata(config, document)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            input_text = f.read()
        print("=" * 50)
        print(f"Processing file: {document['source_path']}")
        print("=" * 50)
    except Exception as e:
        print(f"Error reading input file {input_file}: {e}")
        return None

    all_triples, processed_chunks = process_text_in_chunks(
        config,
        input_text,
        embedding_model,
        debug,
        metadata,
    )

    if not all_triples:
        print(
            f"Knowledge graph generation failed for file: {document['source_path']}. "
            "No triples found."
        )
        return None

    if config.get("standardization", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("PHASE 2: ENTITY STANDARDIZATION")
        print("=" * 50)
        all_triples = standardize_entities(all_triples, config)

    if config.get("inference", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("PHASE 3: RELATIONSHIP INFERENCE")
        print("=" * 50)
        all_triples = infer_relationships(all_triples, config)

    try:
        _save_hybrid_output(output_file, processed_chunks, all_triples, metadata)
        print(f"\nSaved processed hybrid data to {output_file}")
    except Exception as e:
        print(f"Warning: Could not save data to {output_file}: {e}")

    return all_triples


def main_kg_builder(
    input_path,
    output_dir,
    config_path="config.toml",
    experiment_name=None,
    debug=False,
    no_standardize=False,
    no_inference=False,
):
    """Main entry point for the SDG 6 policy knowledge graph generator."""
    config = load_config(config_path)
    if not config:
        print(f"Failed to load configuration from {config_path}. Exiting.")
        return

    if not input_path:
        print("Error: input_path is required.")
        return

    experiment_name = experiment_name or "sdg6_policy_alignment"

    if no_standardize:
        config.setdefault("standardization", {})["enabled"] = False
    if no_inference:
        config.setdefault("inference", {})["enabled"] = False

    os.makedirs(output_dir, exist_ok=True)
    output_folder = os.path.join(output_dir, f"{experiment_name}_json_outputs")
    os.makedirs(output_folder, exist_ok=True)

    try:
        input_documents = _prepare_input_documents(
            config, input_path, output_dir, experiment_name
        )
    except Exception as exc:
        print(f"Error preparing input documents: {exc}")
        return

    if not input_documents:
        print(
            "No supported input documents found. Provide .pdf, .txt, or .md files."
        )
        return

    embedding_config = config.get("embedding", {})
    embedding_model_name = embedding_config.get("model", "all-MiniLM-L6-v2")
    embedding_device = embedding_config.get("device", "auto")
    embedding_model = load_embedding_model(embedding_model_name, embedding_device)
    if embedding_model is None:
        print("Failed to load embedding model. Exiting.")
        return

    processing_config = config.get("processing", {})
    configured_workers = int(processing_config.get("max_workers", 1))
    max_workers = min(max(1, configured_workers), len(input_documents))

    def process_wrapper(index, document):
        print("=" * 50)
        print(
            f"Processing document {index + 1}/{len(input_documents)}: "
            f"{document['source_file']}"
        )
        print("=" * 50)

        output_file_name = f"{document['document_id']}.json"
        output_file_path = os.path.join(output_folder, output_file_name)

        if os.path.exists(output_file_path):
            print(f"Skipping {output_file_name} as it already exists.")
            return

        process_file(config, document, output_file_path, embedding_model, debug)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_wrapper, index, document)
            for index, document in enumerate(input_documents)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred during file processing: {e}")
                traceback.print_exc()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="SDG 6 Policy Alignment Knowledge Graph Generator"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to an SDG 6 PDF/text file or folder containing .pdf, .txt, or .md files",
    )
    parser.add_argument("--output", "-o", required=True, help="Output directory path")
    parser.add_argument(
        "--config", "-c", default="config.toml", help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment_name",
        "-e",
        default="sdg6_policy_alignment",
        help="Name of the experiment for output files",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug output"
    )
    parser.add_argument(
        "--no-standardize", action="store_true", help="Disable entity standardization"
    )
    parser.add_argument(
        "--no-inference", action="store_true", help="Disable relationship inference"
    )
    return parser


def cli():
    parser = build_arg_parser()
    args = parser.parse_args()

    main_kg_builder(
        args.input,
        args.output,
        config_path=args.config,
        experiment_name=args.experiment_name,
        debug=args.debug,
        no_standardize=args.no_standardize,
        no_inference=args.no_inference,
    )


if __name__ == "__main__":
    cli()
