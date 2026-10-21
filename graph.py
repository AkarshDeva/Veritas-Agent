from langgraph.graph import END, START, StateGraph

from config import GOOD_SCORE, MAX_RETRIES
from nodes import (
    abstain_node,
    answer_node,
    grade_node,
    retrieve_node,
    rewrite_node,
    verify_node,
)
from state import AgentState, new_state


# DECISION 1: after grading, where do we go?
def route_after_grade(state: AgentState):
    if state["score"] >= GOOD_SCORE:
        return "write_answer"
    if state["retries"] < MAX_RETRIES:
        return "rewrite"
    return "abstain"


# DECISION 2: after verifying, finish or refuse?
def route_after_verify(state: AgentState):
    if state["verified"]:
        return "done"
    return "abstain"


# Build the map
builder = StateGraph(AgentState)

builder.add_node("search", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("write_answer", answer_node)
builder.add_node("verify", verify_node)
builder.add_node("abstain", abstain_node)

builder.add_edge(START, "search")
builder.add_edge("search", "grade")
builder.add_conditional_edges(
    "grade",
    route_after_grade,
    {"write_answer": "write_answer", "rewrite": "rewrite", "abstain": "abstain"},
)
builder.add_edge("rewrite", "search")  # the retry loop
builder.add_edge("write_answer", "verify")
builder.add_conditional_edges(
    "verify",
    route_after_verify,
    {"done": END, "abstain": "abstain"},
)
builder.add_edge("abstain", END)

graph = builder.compile()


def ask(question: str) -> AgentState:
    """Run the whole agent on one question and return the final state."""
    return graph.invoke(new_state(question))


# Quick test: python graph.py
if __name__ == "__main__":
    tests = [
        "How many days of paid annual leave do employees get?",  # easy, should pass first time
        "Can I work from home?",                                  # vague, may need a rewrite
        "What is the CEO's salary at Nimbus Labs?",              # not in the documents
    ]
    for q in tests:
        result = ask(q)
        print("\n" + "=" * 60)
        print("Q:", q)
        for line in result["trace"]:
            print("  ", line)
        print("ANSWER:", result["answer"])
        print("SOURCES:", result["sources"])
        print("STATUS:", result["status"])

    print("\n--- Mermaid diagram (paste into mermaid.live or your README) ---")
    print(graph.get_graph().draw_mermaid())