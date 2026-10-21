import argparse
import json
import sys
import time
from pathlib import Path

# Let this file import from the project root (graph.py, config.py ...)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from baseline_rag import baseline_answer
from config import CHAT_MODEL
from graph import ask

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "questions.json"
RAW_FILE = HERE / "results.json"
REPORT_FILE = HERE / "results.md"


# ---------- The judge ----------
class Verdict(BaseModel):
    abstained: bool = Field(
        description="True if the system answer says it does not know, cannot find it, "
        "or that the information is not mentioned, instead of giving the requested fact."
    )
    correct: bool = Field(
        description="True if the system answer contains the key facts of the expected answer "
        "and does not contradict it. False if the expected answer is NONE."
    )


JUDGE_PROMPT = """You are marking answers from a document question-answering system.

Question: {question}
Expected answer: {expected}
System answer: {answer}

Fill in the form:
- abstained: did the system decline to answer or say the information is not available?
- correct: does the system answer match the expected answer? Extra correct detail is fine.
  A wrong number or a contradicting fact is NOT correct.
  If the expected answer is NONE (the documents do not contain it), set correct to false.
"""

judge = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0, max_retries=5)
judge = judge.with_structured_output(Verdict)


def with_retry(fn, tries=3):
    """Call fn(); on an error wait and try again (Gemini sometimes returns 503/429)."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as error:
            if attempt == tries - 1:
                raise
            print(f"   (error: {str(error)[:80]} ... retrying)")
            time.sleep(5 * (attempt + 1))


def mark(item, answer):
    """Return the judge's verdict plus the final correct / hallucinated flags."""
    expected = item["expected_answer"] or "NONE"
    prompt = JUDGE_PROMPT.format(question=item["question"], expected=expected, answer=answer)
    verdict = with_retry(lambda: judge.invoke(prompt))
    judge_failed = verdict is None
    abstained = False if judge_failed else verdict.abstained
    judged_correct = False if judge_failed else verdict.correct

    if item["type"] == "unanswerable":
        correct = abstained
        hallucinated = not abstained
    else:
        correct = (not abstained) and judged_correct
        hallucinated = (not abstained) and (not judged_correct)

    return {
        "abstained": abstained,
        "correct": correct,
        "hallucinated": hallucinated,
        "judge_failed": judge_failed,
    }


# ---------- Run both systems ----------
def count_agent_calls(trace):
    """Count Gemini calls the agent made, from its trace lines."""
    calls = 0
    for line in trace:
        if line.startswith(("grade:", "rewrite", "answer:")):
            calls += 1
        elif line.startswith(("verify: supported", "verify: NOT")):
            calls += 1
    return calls


def run_baseline(item):
    start = time.time()
    result = with_retry(lambda: baseline_answer(item["question"]))
    return {"answer": result["answer"], "seconds": time.time() - start, "calls": 1}


def run_agent(item):
    start = time.time()
    result = with_retry(lambda: ask(item["question"]))
    return {
        "answer": result["answer"],
        "seconds": time.time() - start,
        "calls": count_agent_calls(result["trace"]),
        "trace": result["trace"],
    }


# ---------- Report ----------
def rate(flags):
    return f"{100 * sum(flags) / len(flags):.0f}%" if flags else "n/a"


def summarize(rows, system):
    def pick(kind):
        return [r[system] for r in rows if r["type"] == kind]

    everything = [r[system] for r in rows]
    return {
        "Clear questions answered correctly": rate([x["correct"] for x in pick("clear")]),
        "Vague questions answered correctly": rate([x["correct"] for x in pick("vague")]),
        "Unanswerable questions correctly refused": rate([x["correct"] for x in pick("unanswerable")]),
        "Hallucination rate (all questions)": rate([x["hallucinated"] for x in everything]),
        "Average time per question (s)": f"{sum(x['seconds'] for x in everything) / len(everything):.1f}",
        "Average Gemini calls per question": f"{sum(x['calls'] for x in everything) / len(everything):.1f}",
    }


def write_report(rows):
    base = summarize(rows, "baseline")
    agent = summarize(rows, "agent")

    lines = [
        "# Evaluation results",
        "",
        f"Questions evaluated: **{len(rows)}**. Judge: Gemini (`{CHAT_MODEL}`) with structured output.",
        "",
        "| Metric | Naive RAG (baseline) | Veritas Agent |",
        "|---|---|---|",
    ]
    for metric in base:
        lines.append(f"| {metric} | {base[metric]} | {agent[metric]} |")

    failed = [r for r in rows if not r["agent"]["correct"]]
    lines += ["", "## Questions the agent got wrong", ""]
    if not failed:
        lines.append("None.")
    for r in failed:
        lines.append(f"- **#{r['id']} ({r['type']})** {r['question']}")
        lines.append(f"  - Agent said: {r['agent']['answer']}")

    lines += [
        "",
        "## Limitations",
        "",
        "- Small benchmark: a few dozen questions over three synthetic documents.",
        "- The judge is an LLM (the same model family as the system), so some marks may be wrong. "
        "Spot-check `results.json`.",
        "- Time and call counts depend on the API load on the day of the run.",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run only every 4th question (10 questions)")
    parser.add_argument("--pause", type=float, default=0, help="seconds to wait between questions")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if args.quick:
        questions = questions[::4]

    # Resume: load the questions we already finished
    done = {}
    if RAW_FILE.exists():
        done = {row["id"]: row for row in json.loads(RAW_FILE.read_text(encoding="utf-8"))}

    for item in questions:
        if item["id"] in done:
            continue

        row = {"id": item["id"], "type": item["type"], "question": item["question"]}
        for system, runner in (("baseline", run_baseline), ("agent", run_agent)):
            output = runner(item)
            output.update(mark(item, output["answer"]))
            row[system] = output

        done[item["id"]] = row
        RAW_FILE.write_text(json.dumps(list(done.values()), indent=2, ensure_ascii=False), encoding="utf-8")

        tick = lambda ok: "OK " if ok else "BAD"
        print(f"#{item['id']:>2} {item['type']:<12} baseline={tick(row['baseline']['correct'])} "
              f"agent={tick(row['agent']['correct'])}")
        time.sleep(args.pause)

    wanted = {q["id"] for q in questions}
    rows = [done[i] for i in sorted(wanted) if i in done]
    write_report(rows)

    judge_failures = sum(r[s]["judge_failed"] for r in rows for s in ("baseline", "agent"))
    if judge_failures:
        print(f"Warning: the judge failed on {judge_failures} answers (counted as wrong).")
    print(f"\nReport written to {REPORT_FILE}")


if __name__ == "__main__":
    main()