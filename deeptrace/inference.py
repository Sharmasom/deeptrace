"""Standalone inference: video path -> JSON-compatible result + visual artifacts.

Pipeline (identical to training preprocessing):
  validate -> 32 frames (every 10th, first ~10-13 s) -> MTCNN all faces -> IoU tracks (<=3)
  -> 224x224 crops (1.3x margin) -> EfficientNet-B0 features per frame
  -> video score per face track from the 16 even frames (as in training)
  -> frame scores: classifier applied to each frame's own features (exact for the
     'mean' head, because the head is linear: mean of frame logits = video logit)
  -> Grad-CAM on the backbone's last feature map for the most suspicious frames
  -> video verdict = most suspicious face track vs the validation-chosen threshold

Limitations (also shown in the app): scores are model outputs, not calibrated
probabilities of manipulation; heatmaps show where the model's evidence is, not
proof of manipulation or of the method used.

CLI:  python -m deeptrace.inference video.mp4 --weights weights/ --out result.json
"""
import json, time
from pathlib import Path

import cv2
import numpy as np
import torch

from deeptrace.data.prep import sample_frames, track, fill, crop, NUM_FRAMES, DET_MAX_SIDE, DET_THRESH
from deeptrace.models.net import Net

MAX_SECONDS, MIN_FRAMES = 600, 16


class Detector:
    def __init__(self, model_dir, device=None):
        model_dir = Path(model_dir)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.info = json.loads((model_dir / "model_info.json").read_text())
        self.model = Net(self.info["head"], self.info["freq"], pretrained=False)
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.model.to(self.dev).eval()
        self.thr = float(self.info["threshold"])
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=self.dev, post_process=False, select_largest=False,
                           thresholds=[0.6, 0.7, 0.7])

    # ------------------------------------------------------------ steps
    @staticmethod
    def validate(path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("The file could not be opened as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if n <= 0 or w <= 0:
            raise ValueError("The video has no readable frames.")
        dur = n / fps if fps > 0 else 0
        if dur > MAX_SECONDS:
            raise ValueError(f"Video is {dur:.0f} s long; the limit is {MAX_SECONDS} s.")
        return dict(fps=round(fps, 2), frames=n, width=w, height=h, duration_s=round(dur, 2))

    def detect(self, frames, chunk=4):
        """MTCNN in small chunks: a 32-frame batch builds an image pyramid that needs
        >1 GB RAM on CPU (killed on Streamlit Cloud). Chunking gives identical boxes."""
        import gc
        h, w = frames[0].shape[:2]
        s = min(1.0, DET_MAX_SIDE / max(h, w))
        small = [cv2.resize(f, (round(w * s), round(h * s))) for f in frames] if s < 1 else frames
        dets = []
        for i in range(0, len(small), chunk):
            with torch.no_grad():
                boxes, probs, lms = self.mtcnn.detect(np.stack(small[i:i + chunk]), landmarks=True)
            for b, p, l in zip(boxes, probs, lms):
                cur = []
                if b is not None:
                    for bb, pp, ll in zip(b, p, l):
                        if pp is not None and float(pp) >= DET_THRESH:
                            cur.append((np.asarray(bb, dtype=np.float64) / s, np.asarray(ll, dtype=np.float64) / s, float(pp)))
                dets.append(cur)
            gc.collect()
        return dets

    def _features(self, crops, chunk=8):
        feats = []
        with torch.no_grad():
            for i in range(0, len(crops), chunk):
                x = torch.from_numpy(np.stack(crops[i:i + chunk])).permute(0, 3, 1, 2).unsqueeze(0).to(self.dev)
                feats.append(self.model.frame_features(x))                                    # 1,t,D
        return torch.cat(feats, 1)                                                             # 1,T,D

    def score_track(self, crops):
        f = self._features(crops)
        even = f[:, 0::2]                                       # the 16 frames used in training
        with torch.no_grad():
            video_logit = self.model.fc(self.model.head(even)).item()
            frame_logits = self.model.fc(f[0]).squeeze(1).cpu().numpy() if self.info["head"] == "mean" else None
        return 1 / (1 + np.exp(-video_logit)), (None if frame_logits is None else 1 / (1 + np.exp(-frame_logits)))

    def gradcam(self, crop_rgb):
        """Grad-CAM for one frame: d(frame logit)/d(last backbone feature map)."""
        m = self.model
        x = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.dev) / 255
        x01 = x.clone()
        x = (x - m.mean) / m.std
        acts = m.backbone(x)                                    # 1,1280,7,7
        acts.retain_grad()
        f = acts.mean((2, 3))
        if m.freq is not None:
            f = torch.cat([f, m.freq(x01)], 1)
        logit = m.fc(f).sum()
        m.zero_grad(); logit.backward()
        w = acts.grad.mean((2, 3), keepdim=True)
        cam = torch.relu((w * acts).sum(1))[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224)); cam = cam / (cam.max() + 1e-8)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        return (0.55 * crop_rgb + 0.45 * heat).astype(np.uint8)

    # ------------------------------------------------------------ main
    def analyze(self, path, n_cams=4, progress=lambda f, msg: None, out_dir=None):
        t0 = time.time()
        progress(0.05, "Checking the video")
        meta = self.validate(path)
        progress(0.15, "Reading frames")
        frames, idxs, ts = sample_frames(path)
        if len(frames) < MIN_FRAMES:
            raise ValueError("Too few frames could be decoded from this video.")
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1]); idxs.append(idxs[-1]); ts.append(ts[-1])
        progress(0.35, "Finding faces")
        dets = self.detect(frames)
        tracks = track(dets, NUM_FRAMES)
        if not tracks:
            raise ValueError("No face was found in the sampled frames. The detector needs a reasonably visible, frontal face.")
        progress(0.55, "Scoring faces")
        faces = []
        for k, tr in enumerate(tracks):
            seq, filled = fill(tr, NUM_FRAMES)
            crops = [crop(fr, box, lm)[0] for fr, (box, lm) in zip(frames, seq)]
            p_video, p_frames = self.score_track(crops)
            faces.append(dict(face=k, fake_probability=float(p_video), detected_frames=len(tr), crops=crops,
                              frame_scores=None if p_frames is None else [float(v) for v in p_frames],
                              box_first=[float(v) for v in seq[0][0]]))
        top = max(faces, key=lambda f: f["fake_probability"])
        progress(0.75, "Building heatmaps")
        cams = {}
        if top["frame_scores"] is not None:
            order = np.argsort(top["frame_scores"])[::-1][:n_cams]
            for i in order:
                cams[int(i)] = self.gradcam(top["crops"][i])
        p = top["fake_probability"]
        label = "FAKE" if p >= self.thr else "REAL"
        # distance from the threshold on the logit scale, mapped to 0-1 (a ranking aid, not a calibrated probability)
        lg = lambda v: np.log(max(v, 1e-6) / max(1 - v, 1e-6))
        margin = abs(lg(p) - lg(self.thr))
        result = dict(
            predicted_label=label, fake_probability=round(p, 4), real_probability=round(1 - p, 4),
            threshold=self.thr, video_confidence=round(float(1 - np.exp(-margin)), 4),
            n_faces=len(faces), decided_by_face=top["face"],
            frame_timestamps=ts, frame_indices=idxs,
            frame_predictions=None if top["frame_scores"] is None else
                ["FAKE" if v >= self.thr else "REAL" for v in top["frame_scores"]],
            frame_fake_probability=top["frame_scores"],
            faces=[{k: v for k, v in f.items() if k != "crops"} for f in faces],
            heatmap_frames=sorted(cams),
            metadata=dict(video=meta, model=self.info.get("name"), model_info=self.info,
                          device=self.dev, processing_s=round(time.time() - t0, 2)))
        if out_dir:
            out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            paths_ = {}
            for i, img in cams.items():
                fp = out_dir / f"gradcam_frame{i:02d}.jpg"; cv2.imwrite(str(fp), img[:, :, ::-1]); paths_[i] = str(fp)
            result["explainability_artifacts"] = paths_
            (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        progress(1.0, "Done")
        return result, dict(crops=top["crops"], cams=cams, faces=faces)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--weights", default="weights"); ap.add_argument("--out", default="out")
    a = ap.parse_args()
    res, _ = Detector(a.weights).analyze(a.video, out_dir=a.out)
    print(json.dumps({k: v for k, v in res.items() if k not in ("faces",)}, indent=2)[:3000])
