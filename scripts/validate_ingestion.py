import json
from collections import Counter, defaultdict
from pathlib import Path

CHUNK_SIZE = 1000        # should match your Settings
CHUNK_OVERLAP = 200      # should match your Settings

REQUIRED_META_KEYS = ["source_file", "source_path", "page_number", "chunk_index", "chunk_id"]

def iter_jsonl(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            yield line_num, json.loads(line)

def validate(path: str | Path):
    path = Path(path)

    total = 0
    empty_content = 0
    missing_meta = 0
    missing_keys = Counter()
    dup_ids = 0
    chunk_ids = set()
    lengths = []
    per_doc_page_chunks = defaultdict(list)  # (source_file, page_number) -> list of (chunk_index, content)

    for line_num, row in iter_jsonl(path):
        total += 1

        content = row.get("content", "")
        meta = row.get("metadata", {}) or {}

        if not content or not content.strip():
            empty_content += 1

        if not isinstance(meta, dict):
            missing_meta += 1
            continue

        for k in REQUIRED_META_KEYS:
            if k not in meta:
                missing_keys[k] += 1

        cid = meta.get("chunk_id")
        if cid:
            if cid in chunk_ids:
                dup_ids += 1
            chunk_ids.add(cid)

        lengths.append(len(content))
        key = (meta.get("source_file"), meta.get("page_number"))
        per_doc_page_chunks[key].append((meta.get("chunk_index"), content))

    # --- Stats ---
    lengths_sorted = sorted(lengths)
    p50 = lengths_sorted[len(lengths_sorted) // 2] if lengths_sorted else 0
    p95 = lengths_sorted[int(len(lengths_sorted) * 0.95)] if lengths_sorted else 0
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0

    print("\n=== INGESTION VALIDATION REPORT ===")
    print(f"File: {path}")
    print(f"Total chunks: {total}")
    print(f"Empty chunks: {empty_content}")
    print(f"Rows w/ non-dict metadata: {missing_meta}")
    print(f"Duplicate chunk_id: {dup_ids}")
    if missing_keys:
        print("Missing metadata keys counts:", dict(missing_keys))

    print("\n--- Chunk length stats (characters) ---")
    print(f"min={min_len}  p50={p50}  p95={p95}  max={max_len}")
    print(f"Expected chunk_size≈{CHUNK_SIZE} (note: last chunk per page may be smaller)")

    # --- Overlap check (approximate) ---
    # For each (pdf,page), sort by chunk_index and check that the beginning of next chunk
    # roughly matches the end of previous chunk (overlap).
    overlap_misses = 0
    overlap_checks = 0

    def overlap_ok(prev: str, nxt: str, overlap: int) -> bool:
        # Compare suffix(prev, overlap) to prefix(nxt, overlap), forgiving whitespace differences.
        a = prev[-overlap:].strip()
        b = nxt[:overlap].strip()
        if not a or not b:
            return False
        # Exact match is ideal, but PDF text can have whitespace quirks.
        return a == b

    for (src, page), items in per_doc_page_chunks.items():
        items.sort(key=lambda x: (x[0] if x[0] is not None else -1))
        for (idx1, c1), (idx2, c2) in zip(items, items[1:]):
            overlap_checks += 1
            if not overlap_ok(c1, c2, CHUNK_OVERLAP):
                overlap_misses += 1

    print("\n--- Overlap validation (approx) ---")
    print(f"Checks: {overlap_checks} | Mismatches: {overlap_misses}")
    print("Note: Some mismatches are normal with PDFs due to extraction/whitespace. Use as a signal, not absolute truth.")

    # --- Basic pass/fail guidance ---
    print("\n--- Guidance ---")
    if empty_content > 0:
        print("❌ Empty chunks found. Investigate PDF extraction or filtering.")
    if any(missing_keys.values()):
        print("❌ Missing required metadata keys. Fix metadata contract.")
    if dup_ids > 0:
        print("⚠️ Duplicate chunk_id found. This can break upserts later.")
    if total == 0:
        print("❌ No chunks produced.")
    if total > 0 and empty_content == 0 and not any(missing_keys.values()):
        print("✅ Ingestion output looks healthy.")

if __name__ == "__main__":
    validate("ingestion_output/chunks.jsonl")
