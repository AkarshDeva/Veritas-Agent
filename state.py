import operator
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str          # what the user asked (never changes)
    query: str             # what we search with (changes when we rewrite)
    chunks: List[Document] # the pieces of text found by the search
    score: int             # judge's score for the chunks (0 to 10)
    reason: str            # judge's reason for the score          <-- NEW
    retries: int           # how many times we have rewritten the query
    answer: str            # the final answer text
    sources: List[str]     # file names the answer came from
    verified: bool         # did the verifier approve the answer?
    status: str            # "answered", "abstained" or "unverified"
    trace: Annotated[List[str], operator.add]  # step-by-step log (adds, never overwrites)


def new_state(question: str) -> AgentState:
    """The notebook when a new question arrives: everything empty except the question."""
    return {
        "question": question,
        "query": question,
        "chunks": [],
        "score": 0,
        "reason": "",                                            # <-- NEW
        "retries": 0,
        "answer": "",
        "sources": [],
        "verified": False,
        "status": "",
        "trace": [],
    }


# Quick test: python state.py
if __name__ == "__main__":
    print(new_state("How many leave days do I get?"))