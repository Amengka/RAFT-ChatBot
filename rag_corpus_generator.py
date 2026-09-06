import os
import json

INPUT_FOLDER = "./RAG/F1-100" # raw source articles, see RAG/format.txt
OUTPUT_FILE = "raft_documents.json" # name the app reads (chatbot-modified.py:153)
CHUNK_CHAR_LIMIT = 500

def unwrap_delimiters(value, field, filepath):
    """Strip the {...} that RAG/format.txt wraps every source field in.

    The braces are the source layout's field delimiters, not article text. Left in,
    they end up inside the embeddings, inside the keyword matching and inside the LLM
    context (`Title: {Bahrain GP FIA admit...}`). Stripped here rather than downstream
    so chunk boundaries are computed on clean text.

    Only strips a matched pair, so a malformed file is reported instead of silently
    losing its first and last character.
    """
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1].strip()
    print(f"⚠️ {field} is not {{...}}-wrapped, left as is: {filepath}")
    return value

def parse_file(filepath, doc_number):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    if len(lines) < 5:
        print(f"⚠️ Incorrect document format, skipping: {filepath}")
        return []

    source = unwrap_delimiters(lines[0].strip(), "source", filepath)
    title = unwrap_delimiters(lines[1].strip(), "title", filepath)
    author = unwrap_delimiters(lines[2].strip(), "author", filepath)
    date = unwrap_delimiters(lines[3].strip(), "date", filepath)
    content = unwrap_delimiters("\n".join(lines[4:]).strip(), "content", filepath)

    doc_id = f"doc_{doc_number}"
    return split_into_chunks(doc_id, source, title, author, date, content)

def split_into_chunks(doc_id, source, title, author, date, content):
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= CHUNK_CHAR_LIMIT:
            buffer += para + "\n"
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = para + "\n"
    if buffer:
        chunks.append(buffer.strip())

    docs = []
    for i, chunk in enumerate(chunks):
        doc = {
            "id": f"{doc_id}_chunk_{i}",
            "doc_id": doc_id,
            "chunk_index": i,
            "title": title,
            "author": author,
            "date": date,
            "source": source,
            "content": chunk
        }
        docs.append(doc)
    return docs

def process_all_articles(folder):
    all_docs = []
    sorted_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".txt")]
    )

    for idx, filename in enumerate(sorted_files, start=1):
        filepath = os.path.join(folder, filename)
        docs = parse_file(filepath, doc_number=idx)
        all_docs.extend(docs)

    return all_docs

if __name__ == "__main__":
    all_documents = process_all_articles(INPUT_FOLDER)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)

    print(f"✅ 已处理 {len(all_documents)} 个文档段落，结果保存到 {OUTPUT_FILE}")
