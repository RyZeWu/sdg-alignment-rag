# SDG 6 Policy Alignment Knowledge Graph Builder

Prototype pipeline for building a knowledge graph from SDG 6 policy documents. The current workflow is document-folder based: provide a folder of PDFs, text files, or Markdown files, and the builder converts PDFs to text, chunks each document, embeds each chunk, extracts triples with a local Ollama model, and writes hybrid chunk/triple JSON outputs.

## What Changed

- Removed the PubMed-oriented workflow from the active builder path.
- Added direct SDG 6 PDF/text folder ingestion.
- Added SDG policy-alignment extraction prompts.
- Routed graph extraction, entity standardization, and inference through a provider-aware LLM helper.
- Set the default provider to local Ollama for DGX Spark/local GPU use.
- Preserved document provenance on chunks and triples with `document_id`, `source_file`, `source_path`, `source_type`, `domain`, and `sdg_focus`.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Ollama and make sure your chosen model is available:

```bash
ollama serve
ollama list
ollama pull qwen3.6:27b
```

Edit `config.toml` if your local model name differs:

```toml
[llm]
provider = "ollama"
model = "qwen3.6:27b"
base_url = "http://localhost:11434"
```

## Run

From `eGOT/src/graph-builder`:

```bash
python -m src.knowledge_graph_builder.main \
  --input /path/to/sdg6_pdfs \
  --output /path/to/output \
  --config config.toml \
  --experiment_name sdg6_policy_alignment
```

The input path can be one file or a folder containing `.pdf`, `.txt`, or `.md` documents. PDFs are converted to text under:

```text
<output>/sdg6_policy_alignment_text_inputs/
```

Per-document graph JSON is written under:

```text
<output>/sdg6_policy_alignment_json_outputs/
```

## Current Prototype Settings

The current config keeps entity standardization and inference enabled, while reducing runtime with larger chunks and a per-chunk extraction budget:

```toml
[llm]
max_tokens = 1536
temperature = 0.1

[extraction]
max_triples_per_chunk = 25

[chunking]
chunk_size = 1200
overlap = 50

[processing]
max_workers = 1

[standardization]
enabled = true
use_llm_for_entities = true

[inference]
enabled = true
use_llm_for_inference = true
```

`max_workers` is kept at `1` because local Ollama inference on a large model is usually the bottleneck.

## Outputs

Each JSON file contains chunk records with embeddings and triples:

```json
{
  "id": "chunk uuid",
  "chunk_index": 1,
  "text": "...",
  "embedding": [0.01, 0.02],
  "document_id": "example_policy",
  "source_file": "example policy.pdf",
  "source_path": "/path/to/example policy.pdf",
  "source_type": "pdf",
  "domain": "sdg6_policy_alignment",
  "sdg_focus": "sdg 6",
  "triples": [
    {
      "subject": "national water strategy",
      "predicate": "supports",
      "object": "sdg 6.1",
      "chunk_id": "chunk uuid"
    }
  ]
}
```
