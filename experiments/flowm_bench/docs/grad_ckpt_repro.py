# Reproduction: --grad_ckpt with --node_update_pred off changes gradients (adaptive model, lk node update).
# forward()'s finally block resets cell.update_nodes = True; the non-reentrant checkpoint recomputes the
# prediction-phase steps during backward(), after that reset, so the recomputed forward updates nodes while the
# original did not.
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
        pred_frames = m(obs, a_in, a_out)
        loss = (pred_frames - 0.3).square().mean(); loss.backward()
        res.append((loss.item(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
    rel = max((res[0][1][n] - res[1][1][n]).abs().max().item() / (res[0][1][n].abs().max().item() + 1e-12) for n in res[0][1])
    print(f'node_update_pred={pred}: loss(no ckpt)={res[0][0]:.6f} loss(ckpt)={res[1][0]:.6f}  max relative grad difference = {rel:.3e}')
