# Setup — one-time accounts (Modal GPU + Hugging Face)

The Mac does the geometry; MapAnything and SAM 3 run on a [Modal](https://modal.com) GPU.
You need a Modal account with a payment method and a Hugging Face token with access to `facebook/sam3` (gated).

## 1. Hugging Face token (MapAnything weights; SAM 3 is gated: request access at huggingface.co/facebook/sam3 first)
1. Open https://huggingface.co/settings/tokens → **Create new token**.
2. Token type: **Read**. Name it `floorplan-modal`.
3. Copy the token (starts with `hf_`). You won't see it again.

## 2. Put the token into Modal (so GPU jobs can download weights)
1. Open https://modal.com/secrets → **Create new secret**.
2. Pick the **Hugging Face** template (or "Custom").
3. Secret name: **`huggingface-secret`** (exactly; `HF_SECRET` in `fp/recon/hello_gpu.py` uses this name).
4. Key: **`HF_TOKEN`**, Value: the `hf_...` token from step 1.
5. Save.

Check from the repo (after `uv sync`). Free: no container starts.
```bash
uv run python scripts/check_setup.py   # Mac HF token, Modal login, secret has HF_TOKEN, this month's bill
```

## 2b. Add a payment method in Modal (BLOCKER for GPUs)
Modal refuses GPU jobs without a card on file, even with credits
(`Please add a payment method to use A10G GPU functions`).
1. Open https://modal.com/settings → **Usage and Billing** → add a card.
2. Set a **workspace spend limit** on the same page (e.g. $50) so credits can't be overrun.
3. Test:
   - `uv run modal run fp/recon/hello_gpu.py` (CPU only, under $0.01) → prints `hf_user=<you> token_name=... role=read`.
   - `uv run modal run fp/recon/hello_gpu.py --gpu` (A10G, a few cents) → also prints `cuda=True gpu=NVIDIA A10 ...` (Modal's A10G shows as A10).

## 3. Modal login on this machine
```bash
uv sync
uv run modal token new        # opens the browser once; writes ~/.modal.toml
```

## 4. Watch Modal spend (free, read-only)
```bash
uv run python scripts/modal_usage.py           # this month, last month, last 7 days per app
uv run python scripts/modal_usage.py --today   # today per hour
```
Modal reports full intervals only, so a run shows up after its day (or hour) ends.
The credit balance left is only on the dashboard: `uv run modal dashboard` → Usage and Billing.
