"""
dashboard.py — Streamlit A/B test dashboard for NyayaGPT.

Shows side-by-side responses from FP16 and INT4 GGUF NyayaGPT variants.
Routes a random variant label and logs each request to MLflow.

Run:
  streamlit run src/nyaya_pipeline/dashboard.py
"""
import sys
import time
from pathlib import Path

# Add project root to path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False
    st = None  # type: ignore

from nyaya_pipeline import config
from nyaya_pipeline.infer import ab_generate
from nyaya_pipeline.logger import get_logger

log = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
if not _STREAMLIT_AVAILABLE:
    raise ImportError("streamlit is required. pip install streamlit")

st.set_page_config(
    page_title="NyayaGPT A/B Dashboard",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ NyayaGPT — A/B Test Dashboard")
st.caption("FP16 reference vs INT4 deployment candidate on Indian legal questions")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    max_tokens = st.slider("Max new tokens", 64, 512, 256, 32)
    st.divider()
    st.markdown("**MLflow Tracking**")
    st.code(f"URI: {config.MLFLOW_URI}")
    st.markdown("[Open MLflow UI](http://localhost:5000)", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Example questions**")
    examples = [
        "What is Section 302 of the IPC?",
        "Explain the right to life under Article 21.",
        "What constitutes dishonour of cheque under NI Act?",
        "What is a PIL and who can file it?",
        "Explain anticipatory bail under CrPC.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

# ── History ───────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
if "question" not in st.session_state:
    st.session_state["question"] = ""

# ── Input ────────────────────────────────────────────────────────────────────
question = st.text_area(
    "Your legal question:",
    value=st.session_state["question"],
    height=80,
    placeholder="e.g. What are the grounds for anticipatory bail in India?",
)

col_send, col_clear = st.columns([1, 5])
with col_send:
    send = st.button("Send ⚡", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()

# ── Generate ──────────────────────────────────────────────────────────────────
if send and question.strip():
    st.session_state["question"] = ""
    with st.spinner("Generating responses …"):
        try:
            variant, base_resp, ft_resp, base_ms, ft_ms = ab_generate(
                question=question,
                max_new_tokens=max_tokens,
            )
            st.session_state["history"].insert(0, {
                "question": question,
                "variant":  variant,
                "base":     base_resp,
                "ft":       ft_resp,
                "base_ms":  base_ms,
                "ft_ms":    ft_ms,
            })
        except Exception as exc:
            st.error(f"Generation error: {exc}")

# ── Display history ───────────────────────────────────────────────────────────
for entry in st.session_state["history"]:
    st.divider()
    st.markdown(f"**Q:** {entry['question']}")
    st.caption(f"A/B variant served: **{entry['variant']}**")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🔵 NyayaGPT FP16 (Reference)")
        st.info(entry["base"])
        st.caption(f"Latency: {entry['base_ms']:.0f} ms")

    with col_b:
        st.markdown("#### 🟢 NyayaGPT INT4 GGUF")
        st.success(entry["ft"])
        st.caption(f"Latency: {entry['ft_ms']:.0f} ms")

# ── Stats ─────────────────────────────────────────────────────────────────────
if st.session_state["history"]:
    st.divider()
    st.subheader("Session Stats")
    history = st.session_state["history"]
    n       = len(history)
    avg_base = sum(h["base_ms"] for h in history) / n
    avg_ft   = sum(h["ft_ms"]   for h in history) / n
    served_ft = sum(1 for h in history if h["variant"] == "finetuned")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries",         n)
    c2.metric("INT4 served",           f"{served_ft}/{n}")
    c3.metric("Avg FP16 latency",      f"{avg_base:.0f} ms")
    c4.metric("Avg INT4 latency",      f"{avg_ft:.0f} ms",
              delta=f"{avg_ft - avg_base:+.0f} ms")
