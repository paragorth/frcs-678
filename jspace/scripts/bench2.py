"""Cost of restricting source_layers and tuning CPU throughput."""
import sys, time, torch
sys.path.insert(0, "scripts")
from common import load_model, peak_rss_gb
from jlens.fitting import jacobian_for_prompt

model, hf, tok, device, dtype = load_model()
text = ("The history of cartography is the study of how maps have been made "
        "and used across cultures. Early surviving examples are clay tablets "
        "from Mesopotamia showing river courses and field boundaries. ") * 4

# Stage 3 needs: injection at ~25% depth (L6) and readout over the middle
# third (L8-L15). Fitting only these lets backward stop at L6, not L0.
band = [6] + list(range(8, 16))
for threads, dim_batch, seq in ((4, 8, 128), (8, 8, 128), (8, 32, 128), (8, 32, 64)):
    torch.set_num_threads(threads)
    t = time.time()
    jacobian_for_prompt(model, text, band, dim_batch=dim_batch, max_seq_len=seq)
    dt = time.time() - t
    print(f"threads={threads} dim_batch={dim_batch} seq={seq}: {dt:.1f}s/prompt "
          f"-> 100 prompts = {dt*100/3600:.2f} h ; peak rss {peak_rss_gb():.2f} GB",
          flush=True)
