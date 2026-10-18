"""dtype x seq_len sweep. dim_batch=8, threads=4 (both already shown optimal)."""
import sys, time, torch, transformers, jlens
sys.path.insert(0, "scripts")
from common import MODEL_NAME, peak_rss_gb
from jlens.fitting import jacobian_for_prompt

text = ("The history of cartography is the study of how maps have been made "
        "and used across cultures. Early surviving examples are clay tablets "
        "from Mesopotamia showing river courses and field boundaries. ") * 4
band = [6] + list(range(8, 16))
torch.set_num_threads(4)
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    hf.to("cpu").eval()
    model = jlens.from_hf(hf, tok)
    for seq in (48, 128):
        try:
            t = time.time()
            jacobian_for_prompt(model, text, band, dim_batch=8, max_seq_len=seq)
            dt = time.time() - t
            print(f"{str(dtype):16s} seq={seq:<4d} {dt:7.1f}s/prompt "
                  f"-> 100 prompts = {dt*100/3600:5.2f} h ; rss {peak_rss_gb():.2f} GB",
                  flush=True)
        except Exception as e:
            print(f"{str(dtype):16s} seq={seq:<4d} FAILED: {type(e).__name__}: {e}",
                  flush=True)
    del hf, model
