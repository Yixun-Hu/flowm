The small pre-existing behaviour mentioned in #1, with a reproduction.

**Summary.** With `--grad_ckpt` and `--node_update_pred off` on the adaptive model, the gradients differ from the non-checkpointed run (3.0e-2 maximum relative difference in the reproduction below; 0.4–3% depending on configuration in my tests). The loss is identical, and no error is raised. With `--node_update_pred on` (the default, used for the published results) the two runs agree exactly.

**Cause.** `_forward` sets `self.cell.update_nodes = False` before the prediction loop, and `forward`'s `finally` block resets it to `True` on exit. With `torch.utils.checkpoint` (non-reentrant), the prediction-phase steps are recomputed during `backward()`, which runs after that reset, so the recomputed forward updates the nodes while the original forward did not, and the saved activations no longer match the graph the gradients are taken from.

**Reproduction** (CPU, torch 2.9.0, main@302d8f8; run from the repository root):

```python
import sys, torch
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '.')
from kwm.model import build_model
torch.use_deterministic_algorithms(True)
for pred in ('on', 'off'):
    res = []
    for ck in (False, True):
        torch.manual_seed(5)
        m = build_model(dict(model='adaptive', C=4, world=16, window=8, node_update='lk',
                             node_update_pred=pred, grad_ckpt=ck)).train()
        torch.manual_seed(4)
        obs, a_in, a_out = torch.rand(2, 4, 1, 8, 8), torch.randn(2, 4, 2), torch.randn(2, 3, 2)
        loss = (m(obs, a_in, a_out) - 0.3).square().mean(); loss.backward()
        res.append((loss.item(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
    rel = max((res[0][1][n] - res[1][1][n]).abs().max().item() / (res[0][1][n].abs().max().item() + 1e-12) for n in res[0][1])
    print(f'node_update_pred={pred}: loss(no ckpt)={res[0][0]:.6f} loss(ckpt)={res[1][0]:.6f}  max relative grad difference = {rel:.3e}')
```

Output:

```
node_update_pred=on: loss(no ckpt)=0.125550 loss(ckpt)=0.125550  max relative grad difference = 0.000e+00
node_update_pred=off: loss(no ckpt)=0.126230 loss(ckpt)=0.126230  max relative grad difference = 3.010e-02
```

**Possible fix.** Pass the update flag into the checkpointed step as an argument (or capture it in the lambda) instead of reading `self.cell.update_nodes` at recompute time, so the recomputation sees the same value as the original forward. I am happy to send that as a two-line PR if you would like, or leave it to you.

Found during an AI-assisted review (Anthropic Claude) of my fork; reproduction verified by hand.
