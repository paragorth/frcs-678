import sys, time, torch
sys.path.insert(0, "scripts")
from common import load_model, peak_rss_gb
from jlens.fitting import jacobian_for_prompt
model, hf, tok, device, dtype = load_model()
text = ("The history of cartography is the study of how maps have been made "
        "and used across cultures. Early surviving examples are clay tablets "
        "from Mesopotamia showing river courses and field boundaries. ") * 4
band = [6] + list(range(8, 16))
torch.set_num_threads(4)
for dim_batch in (16, 32):
    t = time.time()
    jacobian_for_prompt(model, text, band, dim_batch=dim_batch, max_seq_len=128)
    dt = time.time() - t
    print(f"threads=4 dim_batch={dim_batch}: {dt:.1f}s/prompt -> 100 prompts = "
          f"{dt*100/3600:.2f} h ; peak rss {peak_rss_gb():.2f} GB", flush=True)
