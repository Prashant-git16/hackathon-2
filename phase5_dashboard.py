"""
ZeroDefect AI — Phase 5: Streamlit Supervisor Dashboard  (UPDATED)
===================================================================
Changes from original:
  1. INVALID decision handled everywhere — orange banner, warning messages
  2. "Valid Casting – Checking Defects..." status shown during processing
  3. Casting validator threshold exposed as a sidebar slider
  4. Stats counters exclude INVALID frames (only count actual castings)
  5. Log table shows INVALID rows in orange
  6. Updated decision badges: ACCEPT / REJECT / INVALID
  7. Validation score shown on each uploaded image result

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
import threading
from pathlib import Path
from datetime import datetime
from PIL import Image
import io

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

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }

    [data-testid="stMetric"] {
        background-color: #1e1e2e;
        border: 1px solid #333344;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        color: #ff4b4b;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        font-weight: bold !important;
        color: #a0a0b0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ACCEPT banner */
    .decision-accept {
        background: rgba(40, 167, 69, 0.15);
        color: #2ecc71;
        border: 2px solid #2ecc71;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 3px;
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.2);
    }

    /* REJECT banner */
    .decision-reject {
        background: rgba(220, 53, 69, 0.15);
        color: #ff4b4b;
        border: 2px solid #ff4b4b;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 3px;
        box-shadow: 0 0 15px rgba(255, 75, 75, 0.2);
    }

    /* INVALID banner — orange */
    .decision-invalid {
        background: rgba(255, 140, 0, 0.15);
        color: #ff8c00;
        border: 2px solid #ff8c00;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.6rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 0 15px rgba(255, 140, 0, 0.2);
    }

    /* Scanning / neutral state */
    .decision-scanning {
        background: rgba(100, 100, 120, 0.15);
        color: #aaaacc;
        border: 2px solid #555566;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 2px;
    }

    div[data-testid="stSidebarContent"] { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE INIT ────────────────────────────────────────────────────────
# ── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
if "run_id" not in st.session_state:
    st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
# It is highly likely you will also need these initialized for the dashboard:
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "log_data" not in st.session_state:
    st.session_state.log_data = []
def init_state():
    defaults = {
        "engine"              : None,
        "weights_loaded"      : False,
        "webcam_running"      : False,
        "defect_log"          : [],
        "trend_data"          : [],
        "shift_start"         : datetime.now(),
        "total_inspected"     : 0,    # only valid castings
        "total_defects"       : 0,
        "total_invalid"       : 0,    # non-casting items seen
        "last_decision"       : "—",
        "last_status_message" : "",
        "frame_buffer"        : None,
        "cam_index"           : 0,
        "last_processed_time" : 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/factory.png", width=48)
    st.title("ZeroDefect AI")
    st.caption("Smart Visual Inspection System")
    st.divider()

    st.subheader("Model")
    weights_input = st.text_input(
        "Weights path",
        value="./best.pt",
        help="Path to your trained best.pt file",
    )
    device_choice = st.selectbox("Device", ["cpu", "0 (GPU)"], index=0)
    device = "cpu" if "cpu" in device_choice else "0"

    if st.button("Load Model", type="primary", use_container_width=True):
        if Path(weights_input).exists():
            with st.spinner("Loading model..."):
                st.session_state.engine = DefectInferenceEngine(
                    weights_input, device=device
                )
                st.session_state.weights_loaded = True
            st.success("Model loaded!")
        else:
            st.error(f"File not found:\n{weights_input}")

    if st.session_state.weights_loaded:
        st.success("✅ Model active")
    else:
        st.warning("⚠ No model loaded")

    st.divider()

    st.subheader("Settings")
    conf_thresh = st.slider(
        "Confidence threshold", 0.3, 0.95, 0.65, 0.05,
        help="Min confidence to count a detection (default raised to 0.65)",
    )
    casting_thresh = st.slider(
        "Casting validation threshold", 0.20, 0.80, 0.42, 0.02,
        help="How strict the casting pre-check is. Lower = more permissive.",
    )
    cam_index = st.number_input("Camera index", 0, 10, 0)
    st.session_state.cam_index = int(cam_index)

    # Update validator threshold live
    if st.session_state.engine is not None:
        st.session_state.engine.validator.casting_threshold = casting_thresh

    st.divider()
    st.subheader("Shift info")
    st.caption(f"Started: {st.session_state.shift_start.strftime('%H:%M:%S')}")
    elapsed = datetime.now() - st.session_state.shift_start
    st.caption(f"Elapsed: {str(elapsed).split('.')[0]}")

    if st.button("Reset shift", use_container_width=True):
        for key in ["defect_log", "trend_data"]:
            st.session_state[key] = []
        st.session_state.total_inspected = 0
        st.session_state.total_defects   = 0
        st.session_state.total_invalid   = 0
        st.session_state.shift_start     = datetime.now()
        st.rerun()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def update_stats(result: FrameResult):
    """Update session state counters — INVALID frames don't count as inspections."""
    st.session_state.last_decision       = result.decision
    st.session_state.last_status_message = result.status_message

    if result.decision == "INVALID":
        st.session_state.total_invalid += 1
        # Still log to table
        st.session_state.defect_log.append({
            "Time"      : datetime.now().strftime("%H:%M:%S"),
            "Class"     : "—",
            "Confidence": "—",
            "Decision"  : "INVALID",
            "Latency"   : f"{result.inference_ms:.0f}ms",
        })
        return

    # Valid casting — update inspection counters
    st.session_state.total_inspected += 1
    if result.decision == "REJECT":
        st.session_state.total_defects += 1

    for det in result.detections:
        st.session_state.defect_log.append({
            "Time"      : datetime.now().strftime("%H:%M:%S"),
            "Class"     : det["class"],                 # Changed from det.class_name
            "Confidence": f"{det['confidence']:.1%}",   # Changed from det.confidence
            "Decision"  : result.decision,              # Changed from det.decision
            "Latency"   : f"{result.inference_ms:.0f}ms" # Changed from det.inference_ms
        })
    st.session_state.defect_log = st.session_state.defect_log[-200:]

    n = st.session_state.total_inspected
    d = st.session_state.total_defects
    if n % 5 == 0:
        st.session_state.trend_data.append({
            "time"       : datetime.now().strftime("%H:%M:%S"),
            "defect_rate": round(d / n * 100, 1) if n else 0,
            "total"      : n,
        })


def decision_badge(decision: str) -> str:
    """Return an HTML badge for a given decision string."""
    label_map = {
        "ACCEPT" : "✓ No Defect – ACCEPT",
        "REJECT" : "✗ Defect Found – REJECT",
        "INVALID": "⚠ Invalid Item – Not a Casting",
    }
    css_map = {
        "ACCEPT" : "decision-accept",
        "REJECT" : "decision-reject",
        "INVALID": "decision-invalid",
    }
    label = label_map.get(decision, decision)
    css   = css_map.get(decision, "decision-scanning")
    return f'<div class="{css}">{label}</div>'


def process_uploaded_image(uploaded_file, engine, conf):
    """Run full pipeline on uploaded image. Returns (annotated_rgb, result)."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # ── FIXED LINE ──────────────────────────────────────────────
    result     = engine.predict_frame(img_bgr, conf=conf)
    # ────────────────────────────────────────────────────────────
    
    annotated  = engine.annotate_frame(img_bgr, result)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), result


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


# ── METRICS ROW ───────────────────────────────────────────────────────────────

def render_metrics():
    n = st.session_state.total_inspected
    d = st.session_state.total_defects
    inv = st.session_state.total_invalid
    rate = (d / n * 100) if n else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Castings Inspected", f"{n:,}")
    c2.metric("Defects Found",      f"{d:,}")
    c3.metric("Defect Rate",        f"{rate:.1f}%",
              delta_color="inverse", delta=f"{'↑' if rate > 5 else '↓'}{rate:.1f}%")
    c4.metric("Accepted",           f"{n - d:,}")
    c5.metric("Invalid Items ⚠",   f"{inv:,}",
              help="Frames rejected by casting validator (not casting products)")


# ── TREND CHART ───────────────────────────────────────────────────────────────

def render_trend_chart():
    data = st.session_state.trend_data
    if not data:
        st.info("Trend chart will appear after 5 valid casting inspections.")
        return
    df = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["defect_rate"],
        mode="lines+markers",
        name="Defect rate %",
        line=dict(color="#dc3545", width=2),
        fill="tozeroy",
        fillcolor="rgba(220,53,69,0.1)",
    ))
    fig.add_hline(y=5, line_dash="dash", line_color="orange",
                  annotation_text="5% threshold")
    fig.update_layout(
        title="Defect rate over time (valid castings only)",
        xaxis_title="Time", yaxis_title="Defect rate (%)",
        height=280, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── LOG TABLE ─────────────────────────────────────────────────────────────────

def render_log_table():
    log = st.session_state.defect_log
    if not log:
        st.info("No detections yet.")
        return
    df = pd.DataFrame(log[-50:]).iloc[::-1]

    def color_row(val):
        if val == "REJECT":
            return "background-color: #f8d7da; color: #721c24"
        elif val == "ACCEPT":
            return "background-color: #d4edda; color: #155724"
        elif val == "INVALID":
            return "background-color: #fff3cd; color: #856404"
        return ""

    styled = df.style.applymap(color_row, subset=["Decision"])
    st.dataframe(styled, use_container_width=True, height=300)


# ── SHIFT REPORT ──────────────────────────────────────────────────────────────

def render_shift_report():
    n   = st.session_state.total_inspected
    d   = st.session_state.total_defects
    inv = st.session_state.total_invalid
    elapsed = datetime.now() - st.session_state.shift_start

    report = {
        "Shift start"     : st.session_state.shift_start.strftime("%Y-%m-%d %H:%M:%S"),
        "Duration"        : str(elapsed).split(".")[0],
        "Valid castings inspected": n,
        "Defects found"   : d,
        "Accepted"        : n - d,
        "Defect rate"     : f"{(d/n*100) if n else 0:.2f}%",
        "Invalid items blocked": inv,
        "Throughput"      : f"{(n / max(elapsed.seconds, 1) * 60):.1f} castings/min",
    }

    col1, col2 = st.columns(2)
    items = list(report.items())
    with col1:
        for k, v in items[:4]:
            st.metric(k, v)
    with col2:
        for k, v in items[4:]:
            st.metric(k, v)

    report_json = json.dumps(report, indent=2)
    st.download_button(
        label="Download shift report (JSON)",
        data=report_json,
        file_name=f"shift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )


# ── MAIN TABS ─────────────────────────────────────────────────────────────────
st.title("🏭 ZeroDefect AI — Supervisor Dashboard")
render_metrics()
st.divider()

tab_webcam, tab_upload, tab_trends, tab_report, tab_fewshot = st.tabs([
    "📷 Live Webcam",
    "📁 Image Upload",
    "📈 Trends & Charts",
    "📋 Shift Report",
    "🔬 Few-Shot Demo",
])


# ── TAB 1: LIVE WEBCAM ────────────────────────────────────────────────────────
with tab_webcam:
    col_feed, col_info = st.columns([3, 1])

    with col_feed:
        st.subheader("Live inspection feed")
        frame_placeholder  = st.empty()
        status_placeholder = st.empty()

    with col_info:
        st.subheader("Decision")
        decision_placeholder = st.empty()
        st.subheader("Live stats")
        stats_placeholder = st.empty()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_btn = st.button("▶ Start camera", type="primary",
                              disabled=not st.session_state.weights_loaded,
                              use_container_width=True)
    with btn_col2:
        stop_btn = st.button("⏹ Stop camera", use_container_width=True)

    if not st.session_state.weights_loaded:
        st.warning("Load a model in the sidebar first.")

    if start_btn:
        st.session_state.webcam_running = True
    if stop_btn:
        st.session_state.webcam_running = False

    if st.session_state.webcam_running and st.session_state.engine:
        engine = st.session_state.engine
        cap = cv2.VideoCapture(st.session_state.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        status_placeholder.info("Camera running — press Stop to end.")

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Camera read failed.")
                break

            result    = engine.predict_frame(frame, conf=conf_thresh)
            annotated = engine.annotate_frame(frame, result)

            current_time = time.time()
            if (current_time - st.session_state.last_processed_time) > 2.5:
                update_stats(result)
                st.session_state.last_processed_time = current_time

            frame_placeholder.image(bgr_to_pil(annotated), use_container_width=True)

            # Decision badge
            dec = result.decision
            decision_placeholder.markdown(decision_badge(dec), unsafe_allow_html=True)

            # Status message under badge
            if result.status_message:
                decision_placeholder.markdown(
                    f"<small style='color:#888'>{result.status_message}</small>",
                    unsafe_allow_html=True,
                )

            stats_placeholder.json({
                "Castings"  : st.session_state.total_inspected,
                "Defects"   : st.session_state.total_defects,
                "Invalid"   : st.session_state.total_invalid,
                "Rate"      : f"{(st.session_state.total_defects / max(st.session_state.total_inspected, 1) * 100):.1f}%",
                "Latency ms": f"{result.inference_ms:.1f}",
            })

            time.sleep(1.0 / 15)

        cap.release()
        status_placeholder.info("Camera stopped.")


# ── TAB 2: IMAGE UPLOAD ───────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload images for inspection")
    st.caption("The system will first check if each image is a casting product, then run defect detection.")

    if not st.session_state.weights_loaded:
        st.warning("Load a model in the sidebar first.")
    else:
        uploaded_files = st.file_uploader(
            "Drop images here",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            engine = st.session_state.engine
            cols   = st.columns(min(len(uploaded_files), 3))

            for i, uf in enumerate(uploaded_files):
                annotated_rgb, result = process_uploaded_image(uf, engine, conf_thresh)
                update_stats(result)

                with cols[i % 3]:
                    # Decision badge
                    st.markdown(decision_badge(result.decision), unsafe_allow_html=True)
                    st.image(annotated_rgb, caption=uf.name, use_container_width=True)

                    # Status info below image
                    if result.decision == "INVALID":
                        vr = result.validation_result
                        score_str = f" (casting score: {vr.confidence:.2f})" if vr else ""
                        st.warning(f"⚠ Invalid Item – Not a Casting Product{score_str}")
                    elif result.decision == "REJECT":
                        st.error(f"✗ Defective Product – {len(result.detections)} defect(s) found | {result.inference_ms:.0f}ms")
                    else:
                        st.success(f"✓ No Defects – ACCEPT | {result.inference_ms:.0f}ms")

                    # Show validation score in expander
                    if result.validation_result:
                        with st.expander("Casting validator scores"):
                            st.json(result.validation_result.scores)

            st.divider()
            render_log_table()


# ── TAB 3: TRENDS ─────────────────────────────────────────────────────────────
with tab_trends:
    st.subheader("Defect rate over time")
    render_trend_chart()
    st.subheader("Recent detection log")
    render_log_table()

    n = st.session_state.total_inspected
    d = st.session_state.total_defects
    inv = st.session_state.total_invalid
    if n + inv > 0:
        st.subheader("Inspection outcome distribution")
        fig2 = go.Figure(data=[go.Pie(
            labels=["Accepted", "Rejected (Defect)", "Invalid Items Blocked"],
            values=[n - d, d, inv],
            hole=0.4,
            marker_colors=["#28a745", "#dc3545", "#ff8c00"],
        )])
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig2, use_container_width=True)


# ── TAB 4: SHIFT REPORT ───────────────────────────────────────────────────────
with tab_report:
    st.subheader("Shift summary report")
    render_shift_report()


# ── TAB 5: FEW-SHOT DEMO ──────────────────────────────────────────────────────
with tab_fewshot:
    st.subheader("🔬 Few-Shot Learning Demo")
    st.markdown("""
    Upload 10–20 images of a new product (mix of OK and defective),
    and see the model adapt using prototypical networks — no retraining needed.
    """)
    st.info("Load a model in the sidebar first.")

    col_few1, col_few2 = st.columns(2)
    with col_few1:
        st.markdown("**Step 1: Upload support set (new product)**")
        support_ok  = st.file_uploader("OK samples",     type=["jpg","jpeg","png"], accept_multiple_files=True, key="fs_ok")
        support_def = st.file_uploader("Defect samples", type=["jpg","jpeg","png"], accept_multiple_files=True, key="fs_def")
    with col_few2:
        st.markdown("**Step 2: Upload query images to classify**")
        query_imgs  = st.file_uploader("Query images",   type=["jpg","jpeg","png"], accept_multiple_files=True, key="fs_query")

    if st.button("Run Few-Shot Classification", type="primary",
                 disabled=(not st.session_state.weights_loaded or
                           not support_ok or not support_def or not query_imgs)):
        def extract_features(file_list):
            features = []
            for f in file_list:
                data = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
                img  = cv2.resize(img, (224, 224))
                feat = img.reshape(-1, 3).mean(axis=0).tolist()
                gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                feat.append(edges.mean())
                feat.append(edges.std())
                features.append(feat)
            return np.array(features, dtype=np.float32)

        with st.spinner("Extracting features from support set..."):
            ok_feats  = extract_features(support_ok)
            def_feats = extract_features(support_def)

        proto_ok  = ok_feats.mean(axis=0)
        proto_def = def_feats.mean(axis=0)
        st.success(f"Prototypes built from {len(support_ok)} OK + {len(support_def)} defect samples")

        st.markdown("**Results on query images:**")
        q_cols = st.columns(min(len(query_imgs), 4))

        for i, qf in enumerate(query_imgs):
            data = np.asarray(bytearray(qf.read()), dtype=np.uint8)
            img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
            feat = extract_features([qf])
            dist_ok  = np.linalg.norm(feat[0] - proto_ok)
            dist_def = np.linalg.norm(feat[0] - proto_def)
            pred     = "ok" if dist_ok < dist_def else "defect"
            conf     = dist_def / (dist_ok + dist_def + 1e-6)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dec     = "ACCEPT" if pred == "ok" else "REJECT"

            with q_cols[i % 4]:
                st.image(img_rgb, caption=qf.name, use_container_width=True)
                st.markdown(
                    decision_badge(dec) + f"<br><small style='text-align:center;display:block'>conf {conf:.1%}</small>",
                    unsafe_allow_html=True,
                )