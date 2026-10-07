"""CP0 smoke test: Modal auth + HF secret, and optionally a GPU.

  uv run modal run fp/recon/hello_gpu.py         # CPU only, well under $0.01: HF token works inside Modal
  uv run modal run fp/recon/hello_gpu.py --gpu   # also an A10G ($1.10/h): CUDA works, a few cents"""
import modal

HF_SECRET = "huggingface-secret"  # Modal secret holding HF_TOKEN (SETUP.md §2)

app = modal.App("floorplan-hello")
hf_image = modal.Image.debian_slim().pip_install("huggingface_hub")
gpu_image = hf_image.pip_install("torch")
# No silent fallback: the run stops with a clear error if the secret or its key is missing.
hf_secret = modal.Secret.from_name(HF_SECRET, required_keys=["HF_TOKEN"])


@app.function(image=hf_image, secrets=[hf_secret], timeout=120)
def hf_check() -> str:
    import os
    from huggingface_hub import HfApi
    me = HfApi(token=os.environ["HF_TOKEN"]).whoami()
    tok = me.get("auth", {}).get("accessToken", {})
    return f"hf_user={me['name']} token_name={tok.get('displayName', '?')} role={tok.get('role', '?')}"


@app.function(gpu="A10G", image=gpu_image, timeout=300)
def gpu_check() -> str:
    import torch
    return f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}"


@app.local_entrypoint()
def main(gpu: bool = False):
    print(hf_check.remote())
    print(gpu_check.remote() if gpu else "GPU skipped (add --gpu to test an A10G)")
