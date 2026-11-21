"""DeepTrace - Streamlit front end for the deepfake video detector.
Run locally:  streamlit run app.py      (expects weights/model.pt + weights/model_info.json)
"""
import os, json, tempfile, time
os.environ.setdefault("DT_DRIVE", os.path.join(tempfile.gettempdir(), "dt_drive"))
os.environ.setdefault("DT_LOCAL", os.path.join(tempfile.gettempdir(), "dt_local"))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DeepTrace - Deepfake Video Detector", page_icon="🔍", layout="wide")

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def get_detector():
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    from deeptrace.inference import Detector
    return Detector(WEIGHTS, device="cpu")


def load_info():
    try:
        with open(os.path.join(WEIGHTS, "model_info.json")) as f:
            return json.load(f)
    except Exception:
        return {}


info = load_info()

# ------------------------------------------------------------------ header
st.title("🔍 DeepTrace")
st.caption("Face-manipulation (deepfake) detection for videos · EfficientNet-B0 spatio-temporal detector "
           "trained on FaceForensics++ (c23) and evaluated across datasets on Celeb-DF v2")
st.warning("**Automated prediction, not proof.** The result is a statistical estimate from a model trained on "
           "two research datasets. It can be wrong, especially on video types, compression levels or manipulation "
           "methods it has not seen. Do not use it as the only basis for any decision about a person or a video.",
           icon="⚠️")

with st.sidebar:
    st.header("About the model")
    if info:
        st.markdown(f"**Model:** `{info.get('name', '?')}`  \n**Trained on:** {info.get('train_on', '?')}  \n"
                    f"**Decision threshold:** {info.get('threshold', 0):.3f} (chosen on validation data)")
        if info.get("test_auc"):
            st.markdown("**Test AUC (videos never seen in training):**")
            st.table(pd.DataFrame({"AUC": info["test_auc"]}).round(3))
    st.markdown("**How it works**  \n1. Reads 32 frames (first ~10-13 s)  \n2. Finds and tracks every face (MTCNN)  \n"
                "3. Scores each face with EfficientNet-B0  \n4. The most suspicious face decides the verdict  \n"
                "5. Grad-CAM shows where the evidence is")
    st.markdown("**Limits:** one face-swap/reenactment family of fakes; no audio; heatmaps are not proof of "
                "manipulation; scores are not calibrated probabilities.")

# ------------------------------------------------------------------ upload
up = st.file_uploader("Upload a video (mp4, mov, avi, mkv; up to 100 MB)", type=["mp4", "mov", "avi", "mkv"])
if up is None:
    st.info("Upload a short video with a visible face to begin.")
    st.stop()

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Preview")
    st.video(up)
    st.caption(f"{up.name} · {up.size / 1e6:.1f} MB")

key = f"{up.name}-{up.size}"
with c2:
    st.subheader("Analysis")
    run = st.button("Analyze video", type="primary", use_container_width=True)
    if run or st.session_state.get("key") == key:
        if st.session_state.get("key") != key:
            suffix = os.path.splitext(up.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(up.getbuffer()); path = tmp.name
            bar = st.progress(0.0, text="Starting")
            try:
                det = get_detector()
                res, vis = det.analyze(path, progress=lambda f, m: bar.progress(f, text=m))
                st.session_state.update(key=key, res=res, vis=vis, err=None)
            except ValueError as e:
                st.session_state.update(key=key, res=None, vis=None, err=str(e))
            except Exception as e:
                st.session_state.update(key=key, res=None, vis=None, err=f"Unexpected error: {e!r}")
            finally:
                try: os.unlink(path)
                except OSError: pass
                bar.empty()
        if st.session_state.get("err"):
            st.error(st.session_state["err"])
            if st.button("Try again"):
                st.session_state.pop("key", None); st.rerun()
            st.stop()
        res = st.session_state["res"]
        fake = res["predicted_label"] == "FAKE"
        (st.error if fake else st.success)(f"## Verdict: {'LIKELY FAKE' if fake else 'LIKELY REAL'}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Fake score", f"{res['fake_probability']:.3f}")
        m2.metric("Real score", f"{res['real_probability']:.3f}")
        m3.metric("Confidence", f"{res['video_confidence']:.0%}")
        st.caption(f"Verdict = FAKE when the fake score ≥ {res['threshold']:.3f}. Confidence grows with the distance "
                   f"from that threshold; it is a ranking aid, not a guarantee. Faces found: {res['n_faces']} "
                   f"(verdict decided by face #{res['decided_by_face']}). Processed in "
                   f"{res['metadata']['processing_s']} s on {res['metadata']['device'].upper()}.")

if not st.session_state.get("res") or st.session_state.get("key") != key:
    st.stop()
res, vis = st.session_state["res"], st.session_state["vis"]

# ------------------------------------------------------------------ timeline
st.divider()
st.subheader("Frame-by-frame fake score")
if res["frame_fake_probability"]:
    df = pd.DataFrame({"time (s)": res["frame_timestamps"], "fake score": res["frame_fake_probability"],
                       "threshold": res["threshold"]}).set_index("time (s)")
    st.line_chart(df, color=["#d62728", "#7f7f7f"])
    flagged = [t for t, p in zip(res["frame_timestamps"], res["frame_fake_probability"]) if p >= res["threshold"]]
    st.caption(f"Suspicious frames (score ≥ threshold): {len(flagged)} of {len(res['frame_timestamps'])}"
               + (f" · at {', '.join(f'{t:.1f}s' for t in flagged[:12])}{' …' if len(flagged) > 12 else ''}" if flagged else ""))
else:
    st.info("This model type does not produce per-frame scores.")

if len(res["faces"]) > 1:
    st.subheader("Faces in the video")
    st.dataframe(pd.DataFrame([{"face": f["face"], "fake score": round(f["fake_probability"], 4),
                                "frames detected (of 32)": f["detected_frames"]} for f in res["faces"]]),
                 hide_index=True)

# ------------------------------------------------------------------ heatmaps
st.subheader("Most suspicious frames with Grad-CAM")
st.caption("Red areas influenced the model's fake score most. A heatmap shows where the model looked, "
           "not proof that the region was manipulated.")
cams = vis["cams"]
if cams:
    cols = st.columns(len(cams))
    for col, i in zip(cols, sorted(cams, key=lambda i: -res["frame_fake_probability"][i])):
        col.image(cams[i], caption=f"t={res['frame_timestamps'][i]:.1f}s · score {res['frame_fake_probability'][i]:.3f}",
                  use_container_width=True)

st.subheader("Inspect any frame")
i = st.slider("Frame", 0, len(vis["crops"]) - 1, int(np.argmax(res["frame_fake_probability"] or [0])))
a, b = st.columns(2)
a.image(vis["crops"][i], caption=f"Face crop · t={res['frame_timestamps'][i]:.1f}s", use_container_width=True)
if res["frame_fake_probability"]:
    if i not in cams:
        with st.spinner("Computing heatmap"):
            cams[i] = get_detector().gradcam(vis["crops"][i])
    b.image(cams[i], caption=f"Grad-CAM · fake score {res['frame_fake_probability'][i]:.3f}", use_container_width=True)

# ------------------------------------------------------------------ report
st.divider()
report = {k: v for k, v in res.items()}
report["file_name"] = up.name
report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
report["disclaimer"] = "Automated model output; not a guarantee of authenticity or manipulation."
st.download_button("Download prediction report (JSON)", json.dumps(report, indent=2),
                   file_name=f"deeptrace_{os.path.splitext(up.name)[0]}.json", mime="application/json")
