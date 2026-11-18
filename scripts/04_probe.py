"""GPU part: collect '=' residuals for all single-token problems; linear probes per layer for
product units digit, tens digit, hundreds digit, log-magnitude, and operand units digits.
Also logit-lens rank of the correct answer and of the correct units digit class."""
import numpy as np, torch, json
from common import load, prompt, answer_id, layer_out

tok, model = load()
L = model.config.num_hidden_layers
probs = [(a, b) for a in range(10, 100) for b in range(2, 100) if a * b <= 999]
print("n problems", len(probs))
ids = torch.tensor([tok(prompt(a, b)).input_ids for a, b in probs]).cuda()
acts = np.zeros((L + 1, len(probs), model.config.hidden_size), dtype=np.float16)
num_ids = torch.tensor([answer_id(tok, n) for n in range(1000)]).cuda()
W = model.lm_head.weight[num_ids].float()  # 1000 x d
lens_rank = np.zeros((L + 1, len(probs))); lens_digit = np.zeros((L + 1, len(probs)))
true = np.array([a * b for a, b in probs])
for i in range(0, len(probs), 128):
    with torch.no_grad():
        out = model(ids[i:i + 128], output_hidden_states=True)
    for l, h in enumerate(out.hidden_states):
        x = h[:, -1].float()
        acts[l, i:i + 128] = x.cpu().numpy()
        lg = model.model.norm(x.to(torch.bfloat16)).float() @ W.T  # logit lens over 0..999
        t = torch.tensor(true[i:i + 128]).cuda()
        lens_rank[l, i:i + 128] = (lg > lg.gather(1, t[:, None])).sum(1).cpu().numpy()
        # units digit class via logsumexp over tokens with each units digit
        cls = torch.stack([torch.logsumexp(lg[:, d::10], 1) for d in range(10)], 1)
        lens_digit[l, i:i + 128] = (cls.argmax(1) == t % 10).float().cpu().numpy()
np.save("results/cache/acts_eq_all.npy", acts)
np.savez("results/cache/lens.npz", lens_rank=lens_rank, lens_digit=lens_digit, true=true)
