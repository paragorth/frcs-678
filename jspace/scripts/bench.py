import sys, time, torch
sys.path.insert(0, "scripts")
from common import load_model, peak_rss_gb
import jlens
from jlens.fitting import jacobian_for_prompt

model, hf, tok, device, dtype = load_model()
print(f"device={device} dtype={dtype} n_layers={model.n_layers} d_model={model.d_model}")
print(f"rss after load: {peak_rss_gb():.2f} GB")

text = ("The history of cartography is the study of how maps have been made "
        "and used across cultures. Early surviving examples are clay tablets "
        "from Mesopotamia showing river courses and field boundaries. ") * 4

ids = model.encode(text, max_length=128)
print("seq_len", ids.shape)

t = time.time(); model.forward(ids); print(f"single forward: {time.time()-t:.2f}s")

# time a real fit step on ONE prompt, all 24 source layers
t = time.time()
jac, seq_len, nvalid = jacobian_for_prompt(model, text, list(range(23)), dim_batch=8, max_seq_len=128)
dt = time.time() - t
print(f"jacobian_for_prompt (24 layers, dim_batch=8): {dt:.1f}s  seq_len={seq_len} valid={nvalid}")
print(f"peak rss: {peak_rss_gb():.2f} GB")
print(f"=> extrapolated 100 prompts: {dt*100/3600:.1f} h ; 300 prompts: {dt*300/3600:.1f} h")
