from langchain_google_genai import ChatGoogleGenerativeAI

from baseline_rag import get_index, get_text
from config import CHAT_MODEL, TOP_K
from prompts import (
    ANSWER_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    VERIFY_PROMPT,
    format_chunks,
)
from schemas import AnswerCheck, EvidenceGrade, QueryRewrite
from state import AgentState

ABSTAIN_MESSAGE = "I don't know based on the provided documents."

# One model, three "form-filling" versions of it
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0, max_retries=5)
grader = llm.with_structured_output(EvidenceGrade)
rewriter = llm.with_structured_output(QueryRewrite)
verifier = llm.with_structured_output(AnswerCheck)


# WORKER 1: search the index
def retrieve_node(state: AgentState):
    query = state["query"]
    chunks = get_index().similarity_search(query, k=TOP_K)
    return {
        "chunks": chunks,
        "trace": [f"search: '{query}' -> {len(chunks)} chunks"],
    }


# WORKER 2: judge the evidence
def grade_node(state: AgentState):
    prompt = GRADE_PROMPT.format(
        question=state["question"],
        context=format_chunks(state["chunks"]),
    )
    result = grader.invoke(prompt)

    if result is None:  # the model failed to fill the form
        score, reason = 0, "The judge did not return a valid score."
    else:
        score = max(0, min(10, result.score))  # force into 0..10
        reason = result.reason

    return {
        "score": score,
        "reason": reason,
        "trace": [f"grade: {score}/10 - {reason}"],
    }


# WORKER 3: ask a better question
def rewrite_node(state: AgentState):
    prompt = REWRITE_PROMPT.format(
        question=state["question"],
        query=state["query"],
        reason=state["reason"],
    )
    result = rewriter.invoke(prompt)
    new_query = result.new_query if result else state["question"]

    return {
        "query": new_query,
        "retries": state["retries"] + 1,
        "trace": [f"rewrite #{state['retries'] + 1}: '{new_query}'"],
    }


# WORKER 4: write the answer
def answer_node(state: AgentState):
    prompt = ANSWER_PROMPT.format(
        context=format_chunks(state["chunks"]),
        question=state["question"],
    )
    answer = get_text(llm.invoke(prompt).content).strip()
    sources = sorted({c.metadata.get("source", "unknown") for c in state["chunks"]})

    return {
        "answer": answer,
        "sources": sources,
        "trace": ["answer: written from the chunks"],
    }


# WORKER 5: check the answer against the chunks
def verify_node(state: AgentState):
    answer = state["answer"]

    # If the writer already said "I don't know", there is nothing to check
    if "i don't know" in answer.lower():
        return {
            "verified": True,
            "status": "abstained",
            "sources": [],
            "trace": ["verify: the answer is an honest 'I don't know'"],
        }

    prompt = VERIFY_PROMPT.format(
        context=format_chunks(state["chunks"]),
        question=state["question"],
        answer=answer,
    )
    result = verifier.invoke(prompt)
    supported = bool(result and result.is_supported)  # no result = not trusted
    reason = result.reason if result else "The verifier did not return a result."

    return {
        "verified": supported,
        "status": "answered" if supported else "",
        "trace": [f"verify: {'supported' if supported else 'NOT supported'} - {reason}"],
    }


# WORKER 6: refuse politely
def abstain_node(state: AgentState):
    return {
        "answer": ABSTAIN_MESSAGE,
        "sources": [],
        "verified": True,
        "status": "abstained",
        "trace": ["abstain: not enough trusted evidence, so no answer is given"],
    }


# Quick test: python nodes.py  (makes 2 Gemini calls)
if __name__ == "__main__":
    from state import new_state

    s = new_state("How many days of paid annual leave do employees get?")
    s.update(retrieve_node(s))
    s.update(grade_node(s))
    s.update(answer_node(s))
    s.update(verify_node(s))

    for line in s["trace"]:
        print(line)
    print("\nANSWER:", s["answer"])
    print("SOURCES:", s["sources"])
    print("STATUS:", s["status"])