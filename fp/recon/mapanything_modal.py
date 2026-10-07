"""MapAnything on a Modal GPU: RGB images -> per-view metric depth, intrinsics and camera poses.

Called by `fp run` for photo and video input (fp/recon/mapanything.py). All views go through the
model jointly, so they share one world frame: rooms filmed in one walk are already stitched.
Weights: facebook/map-anything-apache (Apache-2.0), cached in the Modal volume
`floorplan-hf-cache` so only the first run downloads them."""
import modal

from fp.recon.hello_gpu import HF_SECRET

MODEL_ID = "facebook/map-anything-apache"

app = modal.App("floorplan-mapanything")
hf_cache = modal.Volume.from_name("floorplan-hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "libgl1", "libglib2.0-0")
         .pip_install("torch", "torchvision")
         .run_commands("git clone --depth 1 https://github.com/facebookresearch/map-anything /opt/map-anything",
                       "pip install -e /opt/map-anything")
         .env({"HF_HOME": "/cache/hf", "TORCH_HOME": "/cache/torch"})
         .add_local_python_source("fp"))


# Any of these fits 150 views; the list lets Modal start on whichever is free (an L40S queue stalled a run).
GPUS = ["L40S", "A100-40GB", "A100-80GB", "H100"]


@app.function(gpu=GPUS, image=image, volumes={"/cache": hf_cache}, timeout=1800,
              secrets=[modal.Secret.from_name(HF_SECRET, required_keys=["HF_TOKEN"])])
def infer(images: list[bytes]) -> bytes:
    """JPEG bytes in, one compressed .npz out: depth (m), conf, mask, K, T_wc (OpenCV cam2world),
    the processed RGB that depth and K refer to, and the metric scale the model applied."""
    import io
    import os
    import tempfile

    import numpy as np
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, b in enumerate(images):
            paths.append(os.path.join(d, f"{i:04d}.jpg"))
            with open(paths[-1], "wb") as fh:
                fh.write(b)
        views = load_images(paths)
    model = MapAnything.from_pretrained(MODEL_ID).to("cuda")
    hf_cache.commit()
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True)

    def stack(key):
        return np.stack([p[key][0].float().cpu().numpy() for p in preds])

    rgb = stack("img_no_norm")
    rgb = rgb * 255 if rgb.max() <= 1.0 else rgb
    buf = io.BytesIO()
    np.savez_compressed(buf, depth=stack("depth_z")[..., 0].astype(np.float16), conf=stack("conf").astype(np.float16),
                        mask=stack("mask")[..., 0] > 0.5, K=stack("intrinsics"), T_wc=stack("camera_poses"),
                        rgb=rgb.clip(0, 255).astype(np.uint8), scale=stack("metric_scaling_factor"))
    return buf.getvalue()
