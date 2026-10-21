from pydantic import BaseModel, Field


class EvidenceGrade(BaseModel):
    """The judge's score for the retrieved chunks."""

    score: int = Field(
        description="How well the chunks can answer the question, from 0 (useless) to 10 (fully answers it)."
    )
    reason: str = Field(description="One short sentence explaining the score.")


class QueryRewrite(BaseModel):
    """A better version of the user's question for searching."""

    new_query: str = Field(
        description="The question rewritten with clearer, more specific keywords for a document search."
    )


class AnswerCheck(BaseModel):
    """The verifier's decision on the final answer."""

    is_supported: bool = Field(
        description="True only if every claim in the answer is backed by the provided context."
    )
    reason: str = Field(description="One short sentence explaining the decision.")


# Quick test: python schemas.py
if __name__ == "__main__":
    from langchain_google_genai import ChatGoogleGenerativeAI

    from config import CHAT_MODEL

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0, max_retries=5)
    judge = llm.with_structured_output(EvidenceGrade)

    result = judge.invoke(
        "Question: How many leave days are there?\n"
        "Chunks: Employees get 24 days of paid annual leave per year.\n"
        "Score how well the chunks answer the question."
    )
    print(result)
    print("score =", result.score, "| type:", type(result.score))