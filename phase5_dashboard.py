"""
ZeroDefect AI — Phase 5: Streamlit Supervisor Dashboard
=========================================================
Decisions shown throughout:
  ✓  ACCEPT  — valid casting, no defects detected         (green)
  ✗  DEFECT  — valid casting, one or more defects found   (red)
  ⚠  INVALID — not a casting product at all               (orange)

Run:
    streamlit run phase5_dashboard.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import json
from pathlib import Path
from datetime import datetime
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from phase4_inference import DefectInferenceEngine, FrameResult

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZeroDefect AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { padding-top: 0.8rem; background: #0d0d14; }

/* ── Metric cards ───────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    transition: transform 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); }
[data-testid="stMetricValue"] {
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    color: #e0e0ff !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #7070a0 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* ── Decision banners ───────────────────── */
.badge-accept {
    background: linear-gradient(135deg,rgba(30,200,90,.18),rgba(30,200,90,.08));
    color: #2ecc71;
    border: 2px solid #2ecc71;
    padding: 1rem 1.5rem;
    border-radius: 14px;
    font-size: 1.6rem;
    font-weight: 900;
    text-align: center;
    letter-spacing: 2px;
    box-shadow: 0 0 20px rgba(46,204,113,.25);
    margin-bottom: 0.5rem;
}
.badge-defect {
    background: linear-gradient(135deg,rgba(220,50,50,.18),rgba(220,50,50,.08));
    color: #ff4b4b;
    border: 2px solid #ff4b4b;
    padding: 1rem 1.5rem;
    border-radius: 14px;
    font-size: 1.6rem;
    font-weight: 900;
    text-align: center;
    letter-spacing: 2px;
    box-shadow: 0 0 20px rgba(255,75,75,.25);
    margin-bottom: 0.5rem;
}
.badge-invalid {
    background: linear-gradient(135deg,rgba(255,140,0,.18),rgba(255,140,0,.08));
    color: #ff9500;
    border: 2px solid #ff9500;
    padding: 1rem 1.5rem;
    border-radius: 14px;
    font-size: 1.45rem;
    font-weight: 900;
    text-align: center;
    letter-spacing: 1.5px;
    box-shadow: 0 0 20px rgba(255,149,0,.25);
    margin-bottom: 0.5rem;
}
.badge-scanning {
    background: rgba(80,80,120,.15);
    color: #9090c0;
    border: 2px solid #404060;
    padding: 1rem 1.5rem;
    border-radius: 14px;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5rem;
}

/* ── Sub-status line ────────────────────── */
.status-sub {
    font-size: 0.82rem;
    color: #7878a8;
    text-align: center;
    margin-top: 0.25rem;
}

/* ── Invalid detail box ─────────────────── */
.invalid-detail {
    background: rgba(255,140,0,.07);
    border-left: 4px solid #ff9500;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-top: 0.4rem;
    font-size: 0.82rem;
    color: #ffb84d;
}

div[data-testid="stSidebarContent"] { padding-top: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "engine":              None,
        "weights_loaded":      False,
        "webcam_running":      False,
        "defect_log":          [],
        "trend_data":          [],
        "shift_start":         datetime.now(),
        "total_inspected":     0,   # only valid castings
        "total_defects":       0,   # castings with defects
        "total_invalid":       0,   # non-casting items
        "last_decision":       "—",
        "last_status_message": "",
        "cam_index":           0,
        "last_log_time":       0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/factory.png", width=44)
    st.title("ZeroDefect AI")
    st.caption("Iron Casting Visual Inspection")
    st.divider()

    st.subheader("⚙ Model")
    weights_input = st.text_input("Weights path", value="./best.pt",
                                  help="Path to trained best.pt")
    device_choice = st.selectbox("Device", ["cpu", "0 (GPU)"], index=0)
    device = "cpu" if "cpu" in device_choice else "0"

    if st.button("Load Model", type="primary", use_container_width=True):
        if Path(weights_input).exists():
            with st.spinner("Loading YOLO model…"):
                st.session_state.engine = DefectInferenceEngine(
                    weights_input, device=device
                )
                st.session_state.weights_loaded = True
            st.success("✅ Model loaded!")
        else:
            st.error(f"File not found: {weights_input}")

    if st.session_state.weights_loaded:
        st.success("✅ Model active")
        # Show class names from model
        eng = st.session_state.engine
        if eng and hasattr(eng.model, "names"):
            st.caption("Model classes: " +
                       ", ".join(f"{v}" for v in eng.model.names.values()))
    else:
        st.warning("⚠ No model loaded")

    st.divider()
    st.subheader("🎛 Thresholds")
    conf_thresh = st.slider(
        "Detection confidence", 0.20, 0.95, 0.45, 0.05,
        help="Min YOLO confidence to count a detection"
    )
    casting_thresh = st.slider(
        "Casting validation", 0.10, 0.70, 0.42, 0.02,
        help="Composite score below this → INVALID. Lower = more permissive."
    )
    cam_index = st.number_input("Camera index", 0, 10, 0)
    st.session_state.cam_index = int(cam_index)

    if st.session_state.engine is not None:
        st.session_state.engine.validator.casting_threshold = casting_thresh

    st.divider()
    st.subheader("🕐 Shift")
    st.caption(f"Started: {st.session_state.shift_start.strftime('%H:%M:%S')}")
    elapsed = datetime.now() - st.session_state.shift_start
    st.caption(f"Elapsed: {str(elapsed).split('.')[0]}")

    if st.button("🔄 Reset shift", use_container_width=True):
        for k in ["defect_log", "trend_data"]:
            st.session_state[k] = []
        st.session_state.total_inspected = 0
        st.session_state.total_defects   = 0
        st.session_state.total_invalid   = 0
        st.session_state.shift_start     = datetime.now()
        st.rerun()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def decision_badge(decision: str) -> str:
    """Returns styled HTML badge for a decision."""
    MAP = {
        "ACCEPT":  ("badge-accept",  "✓  ACCEPT  —  No Defect Found"),
        "DEFECT":  ("badge-defect",  "✗  DEFECT  —  Defect Detected"),
        "INVALID": ("badge-invalid", "⚠  INVALID  —  Not a Casting Product"),
    }
    css, label = MAP.get(decision, ("badge-scanning", f"⏳  {decision}"))
    return f'<div class="{css}">{label}</div>'


def update_stats(result: FrameResult):
    """Update counters and log — INVALID frames not counted as inspections."""
    st.session_state.last_decision       = result.decision
    st.session_state.last_status_message = result.status_message

    if result.decision == "INVALID":
        st.session_state.total_invalid += 1
        st.session_state.defect_log.append({
            "Time":       datetime.now().strftime("%H:%M:%S"),
            "Class":      "—",
            "Confidence": "—",
            "Decision":   "INVALID",
            "Latency":    f"{result.inference_ms:.0f}ms",
        })
        return

    st.session_state.total_inspected += 1
    if result.decision == "DEFECT":
        st.session_state.total_defects += 1

    if result.detections:
        for det in result.detections:
            st.session_state.defect_log.append({
                "Time":       datetime.now().strftime("%H:%M:%S"),
                "Class":      det["class"].upper(),
                "Confidence": f"{det['confidence']:.1%}",
                "Decision":   result.decision,
                "Latency":    f"{result.inference_ms:.0f}ms",
            })
    else:
        # ACCEPT with no boxes → log one ACCEPT row
        st.session_state.defect_log.append({
            "Time":       datetime.now().strftime("%H:%M:%S"),
            "Class":      "OK",
            "Confidence": "—",
            "Decision":   "ACCEPT",
            "Latency":    f"{result.inference_ms:.0f}ms",
        })

    st.session_state.defect_log = st.session_state.defect_log[-300:]

    n = st.session_state.total_inspected
    d = st.session_state.total_defects
    if n > 0 and n % 5 == 0:
        st.session_state.trend_data.append({
            "time":        datetime.now().strftime("%H:%M:%S"),
            "defect_rate": round(d / n * 100, 1),
            "total":       n,
        })


def process_uploaded_image(uploaded_file, engine: DefectInferenceEngine,
                            conf: float):
    """Decode, run full pipeline, return (annotated_rgb, result)."""
    data    = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not decode image: {uploaded_file.name}")
    result    = engine.predict_frame(img_bgr, conf=conf)
    annotated = engine.annotate_frame(img_bgr, result)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), result


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


# ── METRICS ROW ───────────────────────────────────────────────────────────────
def render_metrics():
    n   = st.session_state.total_inspected
    d   = st.session_state.total_defects
    inv = st.session_state.total_invalid
    rate = (d / n * 100) if n else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔩 Castings Inspected", f"{n:,}")
    c2.metric("✗ Defects Found",       f"{d:,}")
    c3.metric("📊 Defect Rate",
              f"{rate:.1f}%",
              delta=f"{'↑' if rate > 5 else '↓'}{rate:.1f}%",
              delta_color="inverse")
    c4.metric("✓ Accepted",            f"{n - d:,}")
    c5.metric("⚠ Invalid Blocked",    f"{inv:,}",
              help="Rejected by casting validator — not iron casting images")


# ── TREND CHART ───────────────────────────────────────────────────────────────
def render_trend_chart():
    data = st.session_state.trend_data
    if not data:
        st.info("Trend chart appears after 5 valid casting inspections.")
        return
    df  = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["defect_rate"],
        mode="lines+markers",
        name="Defect rate %",
        line=dict(color="#ff4b4b", width=2.5),
        fill="tozeroy", fillcolor="rgba(220,53,69,0.12)",
    ))
    fig.add_hline(y=5, line_dash="dash", line_color="#ff9500",
                  annotation_text="5 % threshold")
    fig.update_layout(
        title="Defect rate over time (valid castings only)",
        xaxis_title="Time", yaxis_title="Defect rate (%)",
        height=280, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a0a0c0"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── LOG TABLE ─────────────────────────────────────────────────────────────────
def render_log_table():
    log = st.session_state.defect_log
    if not log:
        st.info("No inspection records yet.")
        return

    df = pd.DataFrame(log[-60:]).iloc[::-1].reset_index(drop=True)

    def _row_style(val):
        return {
            "DEFECT":  "background-color:#3a1a1a;color:#ff8080",
            "ACCEPT":  "background-color:#1a3a1a;color:#80ff80",
            "INVALID": "background-color:#3a2a00;color:#ffcc55",
        }.get(val, "")

    styled = df.style.applymap(_row_style, subset=["Decision"])
    st.dataframe(styled, use_container_width=True, height=300)


# ── SHIFT REPORT ──────────────────────────────────────────────────────────────
def render_shift_report():
    n   = st.session_state.total_inspected
    d   = st.session_state.total_defects
    inv = st.session_state.total_invalid
    elapsed = datetime.now() - st.session_state.shift_start

    report = {
        "Shift start":              st.session_state.shift_start.strftime("%Y-%m-%d %H:%M:%S"),
        "Duration":                 str(elapsed).split(".")[0],
        "Valid castings inspected": n,
        "Defects found":            d,
        "Accepted":                 n - d,
        "Defect rate":              f"{(d / n * 100) if n else 0:.2f}%",
        "Invalid items blocked":    inv,
        "Throughput":               f"{(n / max(elapsed.seconds, 1) * 60):.1f} castings/min",
    }

    c1, c2 = st.columns(2)
    items  = list(report.items())
    with c1:
        for k, v in items[:4]:
            st.metric(k, v)
    with c2:
        for k, v in items[4:]:
            st.metric(k, v)

    st.download_button(
        "⬇ Download shift report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"shift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )


# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.title("🏭 ZeroDefect AI — Iron Casting Inspection")
render_metrics()
st.divider()

tab_webcam, tab_upload, tab_trends, tab_report, tab_fewshot = st.tabs([
    "📷 Live Feed",
    "📁 Image Upload",
    "📈 Trends & Charts",
    "📋 Shift Report",
    "🔬 Few-Shot Demo",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE WEBCAM FEED
# ══════════════════════════════════════════════════════════════════════════════
with tab_webcam:
    if not st.session_state.weights_loaded:
        st.warning("⚠ Please load a model from the sidebar first.")
    else:
        col_feed, col_info = st.columns([3, 1])

        with col_feed:
            st.subheader("📷 Live Inspection Feed")
            frame_ph  = st.empty()     # image placeholder
            status_ph = st.empty()     # status text

        with col_info:
            st.subheader("Result")
            badge_ph = st.empty()
            st.subheader("Stats")
            stats_ph = st.empty()

        b1, b2 = st.columns(2)
        with b1:
            start_btn = st.button("▶ Start",
                                  type="primary",
                                  use_container_width=True,
                                  disabled=not st.session_state.weights_loaded)
        with b2:
            stop_btn = st.button("⏹ Stop", use_container_width=True)

        if start_btn:
            st.session_state.webcam_running = True
        if stop_btn:
            st.session_state.webcam_running = False

        if st.session_state.webcam_running and st.session_state.engine:
            engine = st.session_state.engine
            cap    = cv2.VideoCapture(st.session_state.cam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                st.error(f"❌ Cannot open camera index {st.session_state.cam_index}.")
                st.session_state.webcam_running = False
            else:
                status_ph.info("🎥 Camera running — press ⏹ Stop to end session.")

                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        status_ph.error("Camera read failed — check connection.")
                        break

                    # ── Run inference ──────────────────────────────────
                    result    = engine.predict_frame(frame, conf=conf_thresh)
                    annotated = engine.annotate_frame(frame, result)

                    # ── Show annotated frame ───────────────────────────
                    frame_ph.image(bgr_to_pil(annotated), use_container_width=True)

                    # ── Decision badge ─────────────────────────────────
                    dec = result.decision
                    badge_html = decision_badge(dec)

                    # Sub-status message
                    sub = result.status_message or ""
                    if result.validation_result and not result.validation_result.is_valid:
                        score = result.validation_result.score
                        sub   = f"{sub} (score: {score:.2f})"

                    badge_html += f'<div class="status-sub">{sub}</div>'

                    # If INVALID, show score detail
                    if dec == "INVALID" and result.validation_result:
                        scores_str = " | ".join(
                            f"{k}: {v}"
                            for k, v in result.validation_result.scores.items()
                        )
                        badge_html += (
                            f'<div class="invalid-detail">'
                            f'Validator scores: {scores_str}</div>'
                        )

                    badge_ph.markdown(badge_html, unsafe_allow_html=True)

                    # ── Live stats JSON ────────────────────────────────
                    n   = st.session_state.total_inspected
                    d   = st.session_state.total_defects
                    inv = st.session_state.total_invalid
                    stats_ph.json({
                        "Castings":   n,
                        "Defects":    d,
                        "Accepted":   n - d,
                        "Invalid":    inv,
                        "Rate":       f"{(d / max(n,1) * 100):.1f}%",
                        "Latency ms": f"{result.inference_ms:.1f}",
                        "Boxes":      len(result.detections),
                    })

                    # ── Update counters every 2.5 s ────────────────────
                    now = time.time()
                    if now - st.session_state.last_log_time > 2.5:
                        update_stats(result)
                        st.session_state.last_log_time = now

                    time.sleep(1.0 / 15)  # ~15 fps

                cap.release()
                status_ph.info("Camera stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("📁 Upload Images for Inspection")
    st.caption(
        "The system first checks if each image is a valid iron casting product "
        "and will mark it **INVALID** if it is not. "
        "Valid castings are then scanned for defects → **ACCEPT** or **DEFECT**."
    )

    if not st.session_state.weights_loaded:
        st.warning("⚠ Load a model from the sidebar first.")
    else:
        uploaded_files = st.file_uploader(
            "Drop casting images here (jpg / png / bmp)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="upload_tab_files",
        )

        if uploaded_files:
            engine  = st.session_state.engine
            n_files = len(uploaded_files)
            n_cols  = min(n_files, 3)
            cols    = st.columns(n_cols)

            for i, uf in enumerate(uploaded_files):
                try:
                    annotated_rgb, result = process_uploaded_image(
                        uf, engine, conf_thresh
                    )
                    update_stats(result)
                except Exception as exc:
                    cols[i % n_cols].error(f"Error processing {uf.name}: {exc}")
                    continue

                with cols[i % n_cols]:
                    # ── Decision badge ─────────────────────────────────
                    st.markdown(decision_badge(result.decision),
                                unsafe_allow_html=True)

                    # ── Annotated image ────────────────────────────────
                    st.image(annotated_rgb,
                             caption=f"{uf.name}  •  {result.inference_ms:.0f} ms",
                             use_container_width=True)

                    # ── Result detail ──────────────────────────────────
                    if result.decision == "INVALID":
                        vr = result.validation_result
                        score_str = f"  (casting score: {vr.score:.2f})" if vr else ""
                        st.warning(
                            f"⚠ **Not a casting product**{score_str}\n\n"
                            f"{result.status_message}"
                        )
                        if vr and vr.scores:
                            with st.expander("Validator detail scores"):
                                st.json(vr.scores)

                    elif result.decision == "DEFECT":
                        n_det = len(result.detections)
                        st.error(
                            f"✗ **DEFECT** — {n_det} defect region{'s' if n_det!=1 else ''} detected"
                        )
                        if result.detections:
                            with st.expander("Detection detail"):
                                st.json([{
                                    "class":      d["class"],
                                    "confidence": f"{d['confidence']:.1%}",
                                    "bbox":       [round(v) for v in d["bbox"]],
                                } for d in result.detections])

                    else:  # ACCEPT
                        st.success("✓ **ACCEPT** — No defects found")

            st.divider()
            render_log_table()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRENDS & CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    st.subheader("Defect rate over time")
    render_trend_chart()

    st.subheader("Recent inspection log")
    render_log_table()

    n   = st.session_state.total_inspected
    d   = st.session_state.total_defects
    inv = st.session_state.total_invalid
    if n + inv > 0:
        st.subheader("Inspection outcome distribution")
        fig2 = go.Figure(data=[go.Pie(
            labels=["Accepted", "Defective", "Invalid (Blocked)"],
            values=[max(0, n - d), d, inv],
            hole=0.42,
            marker_colors=["#2ecc71", "#ff4b4b", "#ff9500"],
            textinfo="label+percent",
        )])
        fig2.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c0c0e0"),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHIFT REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.subheader("📋 Shift Summary Report")
    render_shift_report()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FEW-SHOT DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tab_fewshot:
    st.subheader("🔬 Few-Shot Casting Classification Demo")
    st.markdown("""
    Upload 10–20 images of a **new casting type** (mix of good and defective),
    then query unknown images.  
    The system builds prototypes and classifies without retraining.
    """)

    if not st.session_state.weights_loaded:
        st.info("Load a model in the sidebar first.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Step 1 — Support set**")
        supp_ok  = st.file_uploader("OK / Good samples",
                                    type=["jpg","jpeg","png"],
                                    accept_multiple_files=True, key="fs_ok")
        supp_def = st.file_uploader("Defective samples",
                                    type=["jpg","jpeg","png"],
                                    accept_multiple_files=True, key="fs_def")
    with c2:
        st.markdown("**Step 2 — Query images to classify**")
        query_imgs = st.file_uploader("Query images",
                                      type=["jpg","jpeg","png"],
                                      accept_multiple_files=True, key="fs_q")

    ready = (st.session_state.weights_loaded and
             supp_ok and supp_def and query_imgs)

    if st.button("▶ Run Few-Shot Classification",
                 type="primary", disabled=not ready):

        def _features(file_list):
            out = []
            for f in file_list:
                data = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
                img  = cv2.resize(img, (224, 224))
                bgr  = img.reshape(-1, 3).mean(axis=0).tolist()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ed   = cv2.Canny(gray, 50, 150)
                bgr += [ed.mean(), ed.std()]
                out.append(bgr)
            return np.array(out, dtype=np.float32)

        with st.spinner("Building prototypes…"):
            ok_feat  = _features(supp_ok)
            def_feat = _features(supp_def)

        proto_ok  = ok_feat.mean(axis=0)
        proto_def = def_feat.mean(axis=0)
        st.success(f"Prototypes built: {len(supp_ok)} OK + {len(supp_def)} defective")

        st.markdown("**Classification results:**")
        q_cols = st.columns(min(len(query_imgs), 4))
        for i, qf in enumerate(query_imgs):
            data = np.asarray(bytearray(qf.read()), dtype=np.uint8)
            img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
            feat = _features([qf])
            d_ok  = np.linalg.norm(feat[0] - proto_ok)
            d_def = np.linalg.norm(feat[0] - proto_def)
            dec   = "ACCEPT" if d_ok < d_def else "DEFECT"
            conf  = d_def / (d_ok + d_def + 1e-6)
            with q_cols[i % 4]:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption=qf.name, use_container_width=True)
                st.markdown(
                    decision_badge(dec) +
                    f"<div class='status-sub'>conf {conf:.1%}</div>",
                    unsafe_allow_html=True,
                )