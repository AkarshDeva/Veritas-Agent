from langchain_google_genai import ChatGoogleGenerativeAI

from config import CHAT_MODEL, TOP_K
from ingest import load_index

# 1. The model (temperature 0 = no creativity, same answer every time)
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0, max_retries=5)

# 2. Load the saved index only once, then reuse it
_index = None


def get_index():
    global _index
    if _index is None:
        _index = load_index()
    return _index


# 3. Gemini sometimes returns a list instead of text, so make it plain text
def get_text(content):
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content
    )


# 4. The simple RAG: search, then answer. Nothing else.
def baseline_answer(question):
    chunks = get_index().similarity_search(question, k=TOP_K)

    context = "\n\n".join(c.page_content for c in chunks)
    prompt = (
        "Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    answer = get_text(llm.invoke(prompt).content)
    sources = sorted({c.metadata.get("source", "unknown") for c in chunks})

    return {"answer": answer, "sources": sources, "chunks": chunks}


# 5. Quick test: python baseline_rag.py
if __name__ == "__main__":
    tests = [
        "How many days of paid annual leave do employees get?",  # IN the documents
        "What is the CEO's salary at Nimbus Labs?",              # NOT in the documents
    ]
    for q in tests:
        result = baseline_answer(q)
        print("\nQ:", q)
        print("A:", result["answer"])
        print("Sources:", result["sources"])