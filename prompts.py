GRADE_PROMPT = """You are a strict judge for a document question-answering system.

Decide how well the text chunks below can answer the question.

Scoring guide:
- 9-10: the chunks contain the exact answer.
- 6-8: the chunks contain most of the answer, or it is clear but needs a little reading.
- 3-5: the chunks are on the right topic but do NOT contain the answer.
- 0-2: the chunks are unrelated to the question.

Be strict. If the specific fact asked for is not written in the chunks, the score must be 5 or lower, even if the topic is similar.

Question: {question}

Chunks:
{context}
"""


REWRITE_PROMPT = """A search of company documents did not find good evidence for the question below.
Rewrite the question so a document search is more likely to find the answer.

Rules:
- Use clearer, more specific keywords, and add likely formal terms (for example "annual leave" instead of "time off").
- Keep the same meaning. Do not add facts or guess the answer.
- Do not repeat a previous search query.

Original question: {question}
Previous search query: {query}
Reason the last search was weak: {reason}
"""


ANSWER_PROMPT = """Answer the question using ONLY the context below.

Rules:
- Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: I don't know based on the provided documents.
- Be short and direct.
- After each fact, name its source file in square brackets, for example [1_HR_Policy_Nimbus_Labs.pdf].

Context:
{context}

Question: {question}
"""


VERIFY_PROMPT = """You are a fact-checker. Check the answer against the context.

The answer is supported ONLY IF every claim in it can be found in the context.
If the answer adds any fact, number, or detail that is not in the context, it is NOT supported.

Context:
{context}

Question: {question}

Answer to check:
{answer}
"""


def format_chunks(chunks):
    """Turn a list of chunks into one text block, each labelled with its source file."""
    if not chunks:
        return "(no chunks found)"
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        parts.append(f"[Chunk {i} | source: {source}]\n{chunk.page_content}")
    return "\n\n".join(parts)


# Quick test: python prompts.py  (no model call, no cost)
if __name__ == "__main__":
    from langchain_core.documents import Document

    demo = [Document(page_content="Employees get 24 days of paid annual leave.",
                     metadata={"source": "1_HR_Policy_Nimbus_Labs.pdf"})]
    print(GRADE_PROMPT.format(question="How many leave days?", context=format_chunks(demo)))