"""Open-vocabulary damage detection that runs on this machine: Grounding DINO (text -> boxes) then SAM 2.1
(box -> mask). Both are ungated, Apache-2.0 weights from the Hugging Face hub, so an evaluator's laptop can
download them without an account. (SAM 3 does text -> masks in one model but its weights are gated: a cold
run would need HF approval first; it stays available as --backend modal.)

One text pass finds every class: Grounding DINO takes "crack. water stain. mold. peeling paint. hole in wall."
and labels each box with the phrase it matched. A label that does not name exactly one class ("wall", "paint",
"stain hole") is dropped: the model often boxes the whole wall when asked about things on walls.

Deterministic cache (as work order 05): one .npz per image, key = sha1(model ids + parameters + image bytes),
under <cache>/damage/. A replay gives byte-identical detections."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_ID = "facebook/sam2.1-hiera-small"
CLASSES = {                     # class -> words that name it in Grounding DINO's matched phrase
    "crack": ("crack",),
    "water_stain": ("water stain", "stain", "water"),
    "mold": ("mold", "mould"),
    "peeling_paint": ("peeling", "peel"),
    "hole": ("hole",),
}
TEXT = "crack. water stain. mold. peeling paint. hole in wall."
BOX_THRESHOLD = 0.35            # Grounding DINO's published defaults (box 0.35, text 0.25); not tuned here
TEXT_THRESHOLD = 0.25
MAX_MASK_FRAC = 0.25            # a "defect" covering > 25 % of the image is the wall itself, not damage on it
PARAMS = {"detector": DETECTOR_ID, "segmenter": SEGMENTER_ID, "text": TEXT, "box": BOX_THRESHOLD,
          "text_thr": TEXT_THRESHOLD, "max_mask_frac": MAX_MASK_FRAC, "v": 1}

_models: dict = {}


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    if not _models:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        dev = device()
        _models.update(
            dev=dev, gp=AutoProcessor.from_pretrained(DETECTOR_ID),
            gm=AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_ID).to(dev).eval(),
            sp=Sam2Processor.from_pretrained(SEGMENTER_ID),
            sm=Sam2Model.from_pretrained(SEGMENTER_ID).to(dev).eval())
    return _models


def label_class(phrase: str) -> str | None:
    """Grounding DINO's matched phrase -> one damage class, or None if it names none or several."""
    p = phrase.lower()
    hits = {c for c, words in CLASSES.items() if any(w in p for w in words)}
    if hits == {"water_stain", "mold"} and "mold" in p:    # "mold stain": mould
        hits = {"mold"}
    return hits.pop() if len(hits) == 1 else None


def _infer(rgb: np.ndarray) -> dict:
    """RGB uint8 (H, W, 3) -> {boxes (n,4), scores (n,), classes (n,) str, masks (n, H, W) bool}."""
    import torch
    from PIL import Image
    m = _load()
    img = Image.fromarray(rgb)
    inp = m["gp"](images=img, text=TEXT, return_tensors="pt").to(m["dev"])
    with torch.no_grad():
        out = m["gm"](**inp)
    res = m["gp"].post_process_grounded_object_detection(out, inp.input_ids, threshold=BOX_THRESHOLD,
                                                         text_threshold=TEXT_THRESHOLD,
                                                         target_sizes=[img.size[::-1]])[0]
    keep = [(b, float(s), label_class(l)) for b, s, l in
            zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["text_labels"]) if label_class(l)]
    H, W = rgb.shape[:2]
    empty = {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32),
             "classes": np.zeros(0, "<U16"), "masks": np.zeros((0, H, W), bool)}
    if not keep:
        return empty
    boxes = np.array([k[0] for k in keep], np.float32)
    si = m["sp"](images=img, input_boxes=[boxes.tolist()], return_tensors="pt").to(m["dev"])
    with torch.no_grad():
        so = m["sm"](**si, multimask_output=False)
    masks = m["sp"].post_process_masks(so.pred_masks.cpu(), si["original_sizes"])[0][:, 0].numpy().astype(bool)
    ok = masks.reshape(len(masks), -1).mean(1) <= MAX_MASK_FRAC
    if not ok.any():
        return empty
    return {"boxes": boxes[ok], "scores": np.array([k[1] for k in keep], np.float32)[ok],
            "classes": np.array([k[2] for k in keep], "<U16")[ok], "masks": masks[ok]}


def cache_key(rgb: np.ndarray) -> str:
    h = hashlib.sha1(json.dumps(PARAMS, sort_keys=True).encode())
    h.update(np.ascontiguousarray(rgb).tobytes())
    h.update(str(rgb.shape).encode())
    return h.hexdigest()


def detect(images: list[np.ndarray], cache_dir: Path | None = None, use_cache: bool = True) -> list[dict]:
    """RGB images -> per image {boxes, scores, classes, masks}; replayed from the cache when present."""
    out = []
    for rgb in images:
        path = Path(cache_dir) / "damage" / f"{cache_key(rgb)}.npz" if cache_dir else None
        if path is not None and use_cache and path.is_file():
            z = dict(np.load(path, allow_pickle=False))
            n, H, W = int(z["n"]), int(z["h"]), int(z["w"])
            masks = np.unpackbits(z["masks"])[: n * H * W].reshape(n, H, W).astype(bool)
            out.append({"boxes": z["boxes"], "scores": z["scores"], "classes": z["classes"], "masks": masks})
            continue
        r = _infer(rgb)
        out.append(r)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, boxes=r["boxes"], scores=r["scores"], classes=r["classes"],
                                masks=np.packbits(r["masks"].ravel()), n=len(r["masks"]), h=rgb.shape[0], w=rgb.shape[1])
            tmp.replace(path)
    return out
