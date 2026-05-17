import json
import os
import glob
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- Config ---
JSON_FOLDER_PATH = "/home/exouser/llm-knowledge-graph-builder/data/ultradomain/output/cs/cs_json_outputs"
OUTPUT_DIR = "/home/exouser/llm-knowledge-graph-builder/data/ultradomain/output/cs/cs_csv_outputs"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

COMMON_PREDICATES = {
    "created": "CREATED",
    "has": "HAS",
    "is": "IS",
    "are": "ARE",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Output Files ---
DOC_FILE = os.path.join(OUTPUT_DIR, "documents.csv")
ENTITY_FILE = os.path.join(OUTPUT_DIR, "entities.csv")
CHUNK_FILE = os.path.join(OUTPUT_DIR, "chunks.csv")
REL_FILE = os.path.join(OUTPUT_DIR, "relationships.csv")

# --- Collectors ---
all_entities = {}
all_documents = set()
all_chunks = []
all_relationships = []

# --- Helper ---


def get_embedding(text: str):
    return MODEL.encode(text).tolist() if text.strip() else []


def normalize_predicate(pred):
    """Map predicate to a fixed relationship type, or use RELATED."""
    key = pred.strip().lower().replace(" ", "_")
    if key in COMMON_PREDICATES:
        return COMMON_PREDICATES[key], pred  # (rel_type, original predicate)
    
    if "has" in pred:
        return "HAS", pred
    elif "is" in pred:
        return "IS", pred
    elif "created" in pred:
        return "CREATED", pred
    elif "are" in pred:
        return "ARE", pred

    return "RELATED", pred


# --- Process Files ---
json_files = glob.glob(os.path.join(JSON_FOLDER_PATH, "*.json"))
print(f"Found {len(json_files)} JSON files.")

for file_path in tqdm(json_files, desc="Processing JSON files"):
    filename = os.path.basename(file_path)
    all_documents.add(filename)

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Skipping invalid JSON: {file_path}")
            continue

    for chunk in data:
        chunk_id = chunk.get("id")
        if not chunk_id:
            continue

        if chunk_id != "inferred-triples-group":
            # Normal chunk
            text = chunk.get("text", "")
            embedding = chunk.get("embedding") or get_embedding(text)
            all_chunks.append(
                (chunk_id, text, filename, json.dumps(embedding)))

            for triple in chunk.get("triples", []):
                subj, pred, obj = triple["subject"], triple["predicate"], triple["object"]

                if subj not in all_entities:
                    all_entities[subj] = get_embedding(subj)
                if obj not in all_entities:
                    all_entities[obj] = get_embedding(obj)

                rel_type, norm_pred = normalize_predicate(pred)
                all_relationships.append(
                    (subj, rel_type, norm_pred, obj, filename, chunk_id))

        else:
            # Inferred triples
            for triple in chunk.get("triples", []):
                subj, pred, obj = triple["subject"], triple["predicate"], triple["object"]

                if subj not in all_entities:
                    all_entities[subj] = get_embedding(subj)
                if obj not in all_entities:
                    all_entities[obj] = get_embedding(obj)

                rel_type, norm_pred = normalize_predicate(pred)
                all_relationships.append(
                    (subj, rel_type, norm_pred, obj, filename, "inferred"))

# --- Write CSVs ---
print("Writing CSVs...")

with open(DOC_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["fileName"])
    for d in all_documents:
        writer.writerow([d])

with open(ENTITY_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "embedding"])
    for eid, emb in all_entities.items():
        writer.writerow([eid, json.dumps(emb)])

with open(CHUNK_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text", "fileName", "embedding"])
    for c in all_chunks:
        writer.writerow(c)

with open(REL_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "rel_type", "predicate",
                    "object", "fileName", "chunkId"])
    for r in all_relationships:
        writer.writerow(r)

print("✅ CSV export complete!")
print(f"Files saved in {OUTPUT_DIR}")
