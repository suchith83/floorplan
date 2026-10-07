"""SAM 3 on a Modal GPU: images + damage prompts -> instance masks with scores.

Zero-shot text prompts: no training data, and it generalises to an unseen home.
Weights: facebook/sam3 (gated; access granted to the account behind the HF secret)."""
import modal

from fp.recon.hello_gpu import HF_SECRET

MODEL_ID = "facebook/sam3"
PROMPTS = ["crack", "water stain", "mold", "peeling paint", "hole in wall"]

app = modal.App("floorplan-sam3")
hf_cache = modal.Volume.from_name("floorplan-hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "torchvision", "transformers", "accelerate", "pillow", "opencv-python-headless")
         .env({"HF_HOME": "/cache/hf"})
         .add_local_python_source("fp"))


@app.function(gpu="A10G", image=image, volumes={"/cache": hf_cache}, timeout=1800,
              secrets=[modal.Secret.from_name(HF_SECRET, required_keys=["HF_TOKEN"])])
def detect(images: list[bytes], prompts: list[str] = PROMPTS, threshold: float = 0.4) -> list[dict]:
    """JPEG bytes in. Out: one dict per detection: image index, prompt, score, box (x0,y0,x1,y1 px)
    and the mask as PNG bytes at the image's resolution."""
    import io

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    model = Sam3Model.from_pretrained(MODEL_ID).to("cuda").eval()
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    hf_cache.commit()
    found = []
    for i, b in enumerate(images):
        img = Image.open(io.BytesIO(b)).convert("RGB")
        for prompt in prompts:
            inputs = processor(images=img, text=prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            res = processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist())[0]
            for m, s, bx in zip(res["masks"], res["scores"], res["boxes"]):
                mask = m.cpu().numpy().astype(np.uint8) * 255
                found.append({"image": i, "prompt": prompt, "score": round(float(s), 3),
                              "box": [round(float(v), 1) for v in bx], "mask_png": cv2.imencode(".png", mask)[1].tobytes()})
    return found
