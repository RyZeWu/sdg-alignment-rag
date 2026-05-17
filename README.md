# SDG 6 Policy Alignment Knowledge Graph

This repository is a focused prototype for building and querying a knowledge graph for SDG 6 policy alignment. It adapts eGOT from a biomedical/PubMed workflow into a local-document workflow for SDG 6 PDFs and policy documents.

## Repository Structure

- `src/graph-builder/`: builds the SDG 6 reference knowledge graph from PDFs, text files, or Markdown files.
- `src/graph-builder/data/sdg6_pdfs/`: local SDG 6 source PDF corpus.
- `src/graph-builder/outputs/`: generated text and JSON graph outputs from local runs.
- `src/knowledge_graph_retrieval/`: eGOT retrieval service for graph-of-thought question answering over a Neo4j/vector-backed knowledge graph.

The old benchmark datasets and evaluation scripts have been removed so this repository stays focused on the SDG policy-alignment project.

## Current Workflow

1. Build the SDG 6 reference knowledge graph from the PDF corpus.
2. Inspect the generated triples and provenance.
3. Import the graph JSON into Neo4j/vector indexes.
4. Use eGOT retrieval to answer policy-alignment questions with source-grounded evidence.
5. Extend the system so a new government policy can be checked against the SDG 6 reference graph.

## Build The Graph

From `src/graph-builder`:

```bash
source /home/ryzewu/projects/SDG/eGOT/.venv/bin/activate
python -m src.knowledge_graph_builder.main \
  --input data/sdg6_pdfs \
  --output outputs/sdg6 \
  --config config.toml \
  --experiment_name sdg6_policy_alignment
```

The active graph-builder settings live in `src/graph-builder/config.toml`.

## Attribution

This project is adapted from the original eGOT project by Nihar Sanda, Benjamin M. Gyori, Vito Quaranta, Auroop Ganguly, and Ayan Paul.

Original repository: https://github.com/NNeuralDynamics/eGOT

The current version modifies eGOT for SDG policy-alignment knowledge graph construction and retrieval.
