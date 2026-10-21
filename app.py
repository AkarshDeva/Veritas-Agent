import html
import time
from pathlib import Path

import streamlit as st

import baseline_rag
from baseline_rag import baseline_answer
from config import DOCUMENTS_FOLDER, GOOD_SCORE, INDEX_FOLDER
from graph import graph
from ingest import build_index
from state import new_state

st.set_page_config(page_title="Veritas Agent", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..900;1,9..144,400..900&family=Instrument+Sans:wght@400..700&family=JetBrains+Mono:wght@400..700&display=swap');

:root {
  --paper: #f4eee2; --paper2: #ebe3d2; --ink: #14110f; --red: #d9381e;
  --blue: #1d3fd8; --green: #1f7a4d; --mute: #6f655a; --line: #cfc3ab;
  --serif: 'Fraunces', Georgia, serif; --sans: 'Instrument Sans', system-ui, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, Menlo, monospace;
}
#MainMenu, footer {visibility: hidden;}
.stApp {
  font-family: var(--sans); color: var(--ink); background-color: var(--paper);
  background-image: linear-gradient(rgba(20,17,15,.045) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(20,17,15,.045) 1px, transparent 1px);
  background-size: 32px 32px;
}
.block-container {padding-top: 4rem; max-width: 1240px;}
h1, h2, h3, h4, h5 {font-family: var(--serif); color: var(--ink);}

/* ---------- ticker ---------- */
.ticker {overflow: hidden; background: var(--ink); color: var(--paper); border-radius: 4px; margin-bottom: 1.6rem;}
.ticker .run {display: inline-block; white-space: nowrap; padding: .55rem 0; font: 600 .74rem var(--mono);
  letter-spacing: .14em; animation: marquee 32s linear infinite;}
.ticker b {color: #ff8a6b; padding: 0 1.2rem;}
@keyframes marquee {from {transform: translateX(0);} to {transform: translateX(-50%);}}

/* ---------- hero ---------- */
.hero {display: grid; grid-template-columns: 1.5fr 1fr; gap: 1rem; align-items: center; margin-bottom: 1.6rem;}
.hero .kicker {font: 600 .78rem var(--mono); letter-spacing: .18em; color: var(--red); text-transform: uppercase;}
.hero h1 {font-weight: 900; font-size: clamp(3.4rem, 8.5vw, 7rem); line-height: .92; letter-spacing: -.035em;
  margin: .5rem 0 1rem; padding: 0; animation: rise .9s cubic-bezier(.2,.8,.2,1) backwards;}
.hero h1 em {font-style: italic; font-weight: 600; color: var(--red);}
.hero p {font-size: 1.12rem; line-height: 1.55; max-width: 34rem; color: #3b342d; margin: 0 0 1.1rem;}
.rules {display: flex; flex-wrap: wrap; gap: .6rem;}
.rules span {font: 500 .72rem var(--mono); letter-spacing: .08em; text-transform: uppercase; padding: .35rem .6rem;
  border: 1.5px solid var(--ink); background: var(--paper);}
.rules span i {font-style: normal; color: var(--red); margin-right: .4rem;}

/* 3D stack of sheets */
.scene {perspective: 1100px; height: 300px; display: flex; align-items: center; justify-content: center;}
.stack {position: relative; width: 190px; height: 240px; transform-style: preserve-3d;
  transform: rotateX(58deg) rotateZ(-36deg); animation: drift 7s ease-in-out infinite;}
.sheet {position: absolute; inset: 0; border: 2.5px solid var(--ink); border-radius: 6px; padding: 14px;
  font: 700 .72rem var(--mono); letter-spacing: .12em; box-shadow: 10px 10px 0 rgba(20,17,15,.18);}
.sheet.s1 {background: var(--blue); color: var(--paper); transform: translateZ(0);}
.sheet.s2 {background: var(--paper); transform: translateZ(34px);}
.sheet.s3 {background: var(--red); color: var(--paper); transform: translateZ(68px);}
.sheet.s4 {background: var(--paper); transform: translateZ(102px);}
.sheet.s4 .big {font: 900 2.1rem var(--serif); letter-spacing: -.02em; color: var(--green); border: 3px solid var(--green);
  display: inline-block; padding: 4px 10px; margin-top: 48px; transform: rotate(-8deg);}
@keyframes drift {
  0%, 100% {transform: rotateX(58deg) rotateZ(-36deg) translateZ(0);}
  50% {transform: rotateX(54deg) rotateZ(-28deg) translateZ(18px);}
}
@keyframes rise {from {opacity: 0; transform: translateY(28px) rotateX(-14deg);} to {opacity: 1; transform: none;}}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {background: var(--ink); border-right: 3px solid var(--red);}
[data-testid="stSidebar"] * {color: var(--paper);}
.side-label {font: 600 .7rem var(--mono); letter-spacing: .2em; color: #ff8a6b !important; margin: .4rem 0 .6rem;}
.doc {display: flex; gap: .6rem; font: 500 .74rem var(--mono); padding: .45rem 0; border-bottom: 1px solid #2f2923;
  word-break: break-all;}
.doc b {color: #ff8a6b !important; flex-shrink: 0; white-space: nowrap;}
[data-testid="stSidebar"] .stButton button {
  background: transparent; border: 1px solid #4a4238; border-radius: 0; text-align: left; justify-content: flex-start;
  font: 500 .76rem/1.35 var(--mono); padding: .55rem .7rem; transition: all .2s;}
[data-testid="stSidebar"] .stButton button:hover {border-color: var(--red); background: rgba(217,56,30,.14); transform: translateX(5px);}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {background: #1d1914; border: 1px dashed #5a5147; border-radius: 0;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button, [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {color: var(--ink) !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {background: var(--paper); border-radius: 0;}
[data-testid="stSidebar"] .stButton button p, [data-testid="stSidebar"] .stButton button div {text-align: left !important; width: 100%; justify-content: flex-start;}

/* ---------- ask form ---------- */
[data-testid="stForm"] {border: none; padding: 0;}
div[data-baseweb="input"], div[data-baseweb="base-input"], [data-testid="stTextInputRootElement"], [data-testid="stTextInput"] > div > div {background: transparent !important; background-color: transparent !important; border: none !important; border-radius: 0 !important; box-shadow: none !important;}
[data-testid="stTextInput"] [data-testid="stTextInputRootElement"] {border-bottom: 3px solid var(--ink) !important;}
div[data-baseweb="input"] input {font: 500 1.5rem var(--serif) !important; color: var(--ink) !important; padding: .7rem 0 !important; background: transparent !important;}
[data-testid="stTextInput"] input {font: 500 1.5rem var(--serif) !important; color: var(--ink) !important; padding: .7rem 0 !important; background: transparent !important;}
[data-testid="stTextInput"] input::placeholder {color: #a69a85 !important; font-style: italic;}
[data-testid="stForm"] button {
  background: var(--red); color: var(--paper); border: 2px solid var(--ink); border-radius: 0;
  box-shadow: 5px 5px 0 var(--ink); font: 700 .8rem var(--mono); letter-spacing: .12em; text-transform: uppercase;
  padding: .7rem 1.6rem; transition: transform .12s, box-shadow .12s;}
[data-testid="stForm"] button:hover {transform: translate(-2px,-2px); box-shadow: 8px 8px 0 var(--ink); color: var(--paper); border-color: var(--ink);}
[data-testid="stForm"] button:active {transform: translate(4px,4px); box-shadow: 1px 1px 0 var(--ink);}

/* ---------- case header ---------- */
.case {margin: 1.8rem 0 1.2rem; display: flex; gap: 1rem; align-items: baseline; animation: rise .6s backwards;}
.case .tag {font: 700 .72rem var(--mono); letter-spacing: .16em; background: var(--ink); color: var(--paper); padding: .3rem .6rem; white-space: nowrap;}
.case q {font: italic 600 clamp(1.5rem, 3vw, 2.3rem)/1.15 var(--serif); quotes: none;}

/* ---------- exhibits (3D cards) ---------- */
.exhibit {position: relative; background: #fffdf8; border: 2.5px solid var(--ink); padding: 1.3rem 1.4rem 1.1rem;
  box-shadow: 8px 8px 0 var(--ink); transform-style: preserve-3d;
  transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s; animation: rise .7s cubic-bezier(.2,.8,.2,1) backwards;}
.exhibit.left:hover {transform: perspective(1100px) rotateY(5deg) rotateX(2deg) translateY(-6px); box-shadow: 14px 14px 0 var(--ink);}
.exhibit.right:hover {transform: perspective(1100px) rotateY(-5deg) rotateX(2deg) translateY(-6px); box-shadow: 14px 14px 0 var(--blue);}
.exhibit.agent {border-color: var(--blue); box-shadow: 8px 8px 0 var(--blue); animation-delay: .12s;}
.exh-top {display: flex; align-items: baseline; gap: .8rem; padding-bottom: .7rem; margin-bottom: .9rem; border-bottom: 2px solid var(--ink);}
.exhibit.agent .exh-top {border-color: var(--blue);}
.exh-id {font: 700 .68rem var(--mono); letter-spacing: .18em; color: var(--red);}
.exh-name {font: 800 1.25rem var(--serif);}
.answer {font: 400 1.08rem/1.65 var(--sans); min-height: 4.5rem; margin: .4rem 0 1rem; }
.chips {margin-bottom: .3rem;}
.chip {display: inline-block; font: 500 .7rem var(--mono); padding: .22rem .5rem; margin: 0 .4rem .4rem 0;
  border: 1.5px solid var(--ink); background: var(--paper2);}
.exhibit.agent .chip {border-color: var(--blue); color: var(--blue);}
.foot {font: 500 .72rem var(--mono); color: var(--mute); letter-spacing: .04em; margin-top: .5rem;}
.note {font: italic 400 .9rem var(--serif); color: var(--mute); margin-bottom: .6rem;}

.stamp {position: absolute; right: 1.1rem; top: .85rem; font: 900 1.05rem var(--serif); letter-spacing: .1em;
  padding: .25rem .7rem; border: 3px solid; transform: rotate(-9deg); text-transform: uppercase;
  animation: slam .55s cubic-bezier(.2,1.7,.4,1) .45s backwards; mix-blend-mode: multiply;}
.stamp.ok {color: var(--green);} .stamp.no {color: var(--red);} .stamp.grey {color: var(--mute);}
@keyframes slam {from {opacity: 0; transform: scale(3.2) rotate(-26deg);} to {opacity: 1; transform: scale(1) rotate(-9deg);}}

/* ---------- trail ---------- */
.step {display: grid; grid-template-columns: 2.8rem 1fr; padding: .6rem 0; border-bottom: 1.5px dashed var(--line);}
.step:last-child {border-bottom: none;}
.step .num {font: 800 1.35rem var(--mono); color: var(--red); line-height: 1.2;}
.step .label {font: 700 .74rem var(--mono); letter-spacing: .16em; text-transform: uppercase;}
.step .detail {font: 400 .88rem/1.4 var(--sans); color: #4a4137; word-break: break-word; margin-top: .15rem;}
.step.warn .num {color: #b9770e;} .step.bad .num {color: var(--red);} .step.good .num {color: var(--green);}
.step.new {animation: slidein .45s cubic-bezier(.2,.8,.2,1) backwards;}
.step.live .num {color: var(--blue); animation: blink 1s steps(2) infinite;}
@keyframes slidein {from {opacity: 0; transform: translateX(-26px);} to {opacity: 1; transform: none;}}
@keyframes blink {50% {opacity: .15;}}

/* ---------- ledger panels ---------- */
.panel {background: var(--paper2); border: 2px solid var(--ink); padding: 1rem 1.2rem; height: 100%; animation: rise .8s .2s backwards;}
.panel h4 {font: 700 .7rem var(--mono); letter-spacing: .2em; text-transform: uppercase; margin: 0 0 .8rem; color: var(--mute);}
.gauge {text-align: center;}
.gauge svg {width: 100%; max-width: 230px; overflow: visible;}
.gauge .track {fill: none; stroke: var(--line); stroke-width: 12; stroke-linecap: round;}
.gauge .val {fill: none; stroke-width: 12; stroke-linecap: round; stroke-dasharray: var(--len);
  stroke-dashoffset: var(--to); animation: fill 1.4s cubic-bezier(.2,.8,.2,1) .3s backwards;}
@keyframes fill {from {stroke-dashoffset: var(--len);} to {stroke-dashoffset: var(--to);}}
.gauge .score {font: 900 2.6rem var(--serif); margin-top: -1.9rem;}
.gauge .score small {font: 500 .9rem var(--mono); color: var(--mute);}
.bars .row {margin-bottom: .9rem;}
.bars .lab {display: flex; justify-content: space-between; font: 600 .7rem var(--mono); letter-spacing: .05em; margin-bottom: .25rem;}
.bars .track {height: 14px; background: var(--paper); border: 1.5px solid var(--ink);}
.bars .fill {height: 100%; transform-origin: left; animation: grow 1s cubic-bezier(.2,.8,.2,1) .35s backwards;}
.bars .fill.a {background: var(--ink);} .bars .fill.b {background: var(--blue);}
@keyframes grow {from {transform: scaleX(0);} to {transform: scaleX(1);}}
.verdict {font: italic 500 .95rem/1.4 var(--serif); margin-top: .4rem;}

.empty {border: 2px dashed var(--ink); padding: 2.4rem 1.4rem; text-align: center; font: italic 500 1.3rem var(--serif); background: rgba(255,253,248,.6);}
.empty small {display: block; font: 500 .74rem var(--mono); letter-spacing: .12em; margin-top: .7rem; color: var(--mute); font-style: normal;}

@media (max-width: 900px) {.hero {grid-template-columns: 1fr;} .scene {height: 220px;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLES = [
    "How many days of paid annual leave do employees get?",
    "Can I work from home?",
    "What happens if my laptop gets stolen?",
    "The water from my purifier tastes strange.",
    "What is the CEO's salary at Nimbus Labs?",
    "Can the AquaPure X200 purify hot water?",
]
LABELS = {"search": "Search", "grade": "Grade", "rewrite": "Rewrite", "answer": "Write", "verify": "Verify", "abstain": "Refuse"}


def esc(text):
    return html.escape(str(text)).replace("\n", "<br>")


def chips(names):
    return "".join(f'<span class="chip">{esc(n)}</span>' for n in names) or '<span class="foot">no sources cited</span>'


def count_calls(trace):
    calls = 0
    for line in trace:
        if line.startswith(("grade:", "rewrite", "answer:")):
            calls += 1
        elif line.startswith(("verify: supported", "verify: NOT")):
            calls += 1
    return calls


def steps_html(lines, running=False):
    out = []
    for i, line in enumerate(lines, 1):
        head, _, detail = line.partition(":")
        kind = head.split()[0] if head.split() else "search"
        cls = ""
        if kind == "abstain" or "NOT supported" in detail:
            cls = "bad"
        elif kind == "rewrite":
            cls = "warn"
        elif kind == "verify" and "supported" in detail:
            cls = "good"
        if i == len(lines):
            cls += " new"
        out.append(
            f'<div class="step {cls}"><span class="num">{i:02d}</span><div>'
            f'<div class="label">{LABELS.get(kind, kind.title())}</div>'
            f'<div class="detail">{esc(detail.strip())}</div></div></div>'
        )
    if running:
        out.append('<div class="step live"><span class="num">..</span><div><div class="label">Working</div></div></div>')
    return "".join(out)


def live_card(lines, running=True):
    return (
        '<div class="exhibit agent right"><div class="exh-top"><span class="exh-id">EXHIBIT B</span>'
        '<span class="exh-name">Veritas Agent at work</span></div>' + steps_html(lines, running) + "</div>"
    )


def baseline_card(base, seconds):
    return (
        '<div class="exhibit left"><div class="exh-top"><span class="exh-id">EXHIBIT A</span>'
        '<span class="exh-name">Naive RAG</span></div><div class="stamp grey">Unchecked</div>'
        f'<div class="answer">{esc(base["answer"])}</div><div class="chips">{chips(base["sources"])}</div>'
        f'<div class="foot">{seconds:.1f} s &nbsp;/&nbsp; 1 model call &nbsp;/&nbsp; no grading, no verification</div></div>'
    )


def agent_card(final, seconds):
    answered = final["status"] == "answered"
    stamp = '<div class="stamp ok">Verified</div>' if answered else '<div class="stamp no">Withheld</div>'
    note = "" if answered else '<div class="note">No trusted evidence was found, so the agent will not make something up.</div>'
    return (
        '<div class="exhibit agent right"><div class="exh-top"><span class="exh-id">EXHIBIT B</span>'
        f'<span class="exh-name">Veritas Agent</span></div>{stamp}{note}'
        f'<div class="answer">{esc(final["answer"])}</div><div class="chips">{chips(final["sources"])}</div>'
        f'<div class="foot">{seconds:.1f} s &nbsp;/&nbsp; {count_calls(final["trace"])} model calls &nbsp;/&nbsp; {final["retries"]} rewrites</div></div>'
    )


def gauge_html(score):
    length = 157.08
    to = length * (1 - score / 10)
    color = "var(--green)" if score >= GOOD_SCORE else "var(--red)"
    return (
        '<div class="panel gauge"><h4>Evidence score</h4>'
        '<svg viewBox="0 0 120 70"><path class="track" d="M10 62 A50 50 0 0 1 110 62"/>'
        f'<path class="val" style="--len:{length};--to:{to:.2f};stroke:{color}" d="M10 62 A50 50 0 0 1 110 62"/></svg>'
        f'<div class="score">{score}<small> / 10</small></div></div>'
    )


def bars_html(base_s, agent_s, base_c, agent_c):
    top_s, top_c = max(base_s, agent_s, 0.1), max(base_c, agent_c, 1)

    def row(label, a, b, fmt):
        return (
            f'<div class="row"><div class="lab"><span>{label}</span></div>'
            f'<div class="lab"><span>Naive</span><span>{fmt(a[0])}</span></div>'
            f'<div class="track"><div class="fill a" style="width:{100 * a[0] / a[2]:.0f}%"></div></div>'
            f'<div class="lab" style="margin-top:.4rem"><span>Veritas</span><span>{fmt(b[0])}</span></div>'
            f'<div class="track"><div class="fill b" style="width:{100 * b[0] / b[2]:.0f}%"></div></div></div>'
        )

    return (
        '<div class="panel bars"><h4>Cost of being careful</h4>'
        + row("TIME", (base_s, 0, top_s), (agent_s, 0, top_s), lambda v: f"{v:.1f} s")
        + row("MODEL CALLS", (base_c, 0, top_c), (agent_c, 0, top_c), lambda v: f"{v:.0f}")
        + f'<div class="verdict">Safer answers cost {agent_c - base_c:+d} calls and {agent_s - base_s:+.1f} s here.</div></div>'
    )


def run_agent_live(question, placeholder):
    start_state = new_state(question)
    final = new_state(question)
    for update in graph.stream(start_state, stream_mode="updates"):
        for _node, changes in update.items():
            for key, value in changes.items():
                if key == "trace":
                    final["trace"] = final["trace"] + value
                else:
                    final[key] = value
        placeholder.markdown(live_card(final["trace"]), unsafe_allow_html=True)
    return final


def set_question(text):
    st.session_state["question"] = text
    st.session_state["run_now"] = True


st.session_state.setdefault("question", "")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="side-label">LIBRARY</div>', unsafe_allow_html=True)
    files = sorted(p.name for p in Path(DOCUMENTS_FOLDER).glob("*.pdf"))
    st.markdown(
        "".join(f'<div class="doc"><b>{i:02d}</b><span>{esc(n)}</span></div>' for i, n in enumerate(files, 1))
        or '<div class="doc">No PDFs yet</div>',
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Add and rebuild index"):
        Path(DOCUMENTS_FOLDER).mkdir(exist_ok=True)
        for up in uploads:
            (Path(DOCUMENTS_FOLDER) / Path(up.name).name).write_bytes(up.getbuffer())
        with st.spinner("Rebuilding the index..."):
            build_index()
            baseline_rag._index = None
        st.success("Index rebuilt. Refresh the page.")

    st.markdown('<div class="side-label" style="margin-top:1.4rem">SAMPLE QUESTIONS</div>', unsafe_allow_html=True)
    for i, example in enumerate(EXAMPLES):
        st.button(example, key=f"ex{i}", on_click=set_question, args=(example,), use_container_width=True)

# ---------- Header ----------
ticker_text = "".join(
    f"GRADES ITS OWN EVIDENCE<b>/</b>REWRITES WEAK QUERIES<b>/</b>VERIFIES EVERY ANSWER<b>/</b>REFUSES TO GUESS<b>/</b>"
    for _ in range(4)
)
st.markdown(
    f'<div class="ticker"><div class="run">{ticker_text}</div></div>'
    '<div class="hero"><div><div class="kicker">Self-correcting retrieval agent</div>'
    '<h1>Veritas<br><em>Agent</em></h1>'
    "<p>Most RAG systems answer even when they have no evidence. This one checks what it found, tries again when it is "
    "weak, and says it does not know when the documents are silent.</p>"
    '<div class="rules"><span><i>01</i>Grade</span><span><i>02</i>Rewrite</span><span><i>03</i>Verify</span>'
    "<span><i>04</i>Refuse</span></div></div>"
    '<div class="scene"><div class="stack">'
    '<div class="sheet s1">EVIDENCE</div><div class="sheet s2">QUERY</div>'
    '<div class="sheet s3">ANSWER</div><div class="sheet s4"><span class="big">VERIFIED</span></div>'
    "</div></div></div>",
    unsafe_allow_html=True,
)

if not Path(INDEX_FOLDER).exists():
    st.error("No index found. Put PDFs in the documents folder and run:  python ingest.py")
    st.stop()

# ---------- Ask ----------
with st.form("ask_form"):
    st.text_input(
        "Question",
        key="question",
        placeholder="Ask your documents anything...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Run the check")

run = submitted or st.session_state.pop("run_now", False)
question = st.session_state.get("question", "").strip()

if not (run and question):
    st.markdown(
        '<div class="empty">Pick a sample question, or write your own.'
        "<small>TRY THE CEO SALARY QUESTION TO SEE THE DIFFERENCE</small></div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    st.markdown(f'<div class="case"><span class="tag">QUESTION</span><q>{esc(question)}</q></div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        with st.spinner("Naive RAG is answering..."):
            t0 = time.time()
            base = baseline_answer(question)
            base_seconds = time.time() - t0
        st.markdown(baseline_card(base, base_seconds), unsafe_allow_html=True)

    with right:
        box = st.empty()
        t0 = time.time()
        final = run_agent_live(question, box)
        agent_seconds = time.time() - t0
        box.markdown(agent_card(final, agent_seconds), unsafe_allow_html=True)

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns([1, 1.2, 1.6], gap="large")
    p1.markdown(gauge_html(final["score"]), unsafe_allow_html=True)
    p2.markdown(
        bars_html(base_seconds, agent_seconds, 1, count_calls(final["trace"])), unsafe_allow_html=True
    )
    p3.markdown(
        '<div class="panel"><h4>Decision trail</h4>' + steps_html(final["trace"]).replace(" new", "") + "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Evidence the agent used (last search)"):
        for chunk in final["chunks"]:
            st.markdown(f'<span class="chip">{esc(chunk.metadata.get("source", "unknown"))}</span>', unsafe_allow_html=True)
            st.caption(chunk.page_content)

except Exception as error:
    st.error("Something went wrong. Gemini may be busy, so wait a moment and try again.")
    st.caption(str(error)[:300])