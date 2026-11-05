"""Production KWM measurements and its exact untrained lock recurrence."""
from contextlib import contextmanager
import torch
import kwm.lock_nodes as lock_nodes
from kwm.collision import named_nodes
from kwm.rot import rotation_perms, rot_vec
from kwm.transport import masked_mean
from .geometry import BOX, cone_mask, camera_action, turn_k

EVIDENCE_FLOOR = 1e-6
LOCK_AGREEMENT = 0.1


@contextmanager
def agreement_diagnostic():
    """Observe KWM's first coherence call without changing any calculation.

    The API returns raw selected evidence, not its dimensionless LK agreement.
    This scoped hook recovers the denominator used by the production lock test.
    No KWM files are changed. This runner is single-threaded at Python level.
    """
    original = lock_nodes.coherence
    captured = {}

    def diagnostic(m, current, trace, displacement, support=None, *args, **kwargs):
        weight = original(m, current, trace, displacement, support, *args, **kwargs)
        if not captured:
            tr_rel = args[1] if len(args) > 1 else kwargs.get("tr_rel", 0.1)
            mean_support = kwargs.get("mean_support")
            ref = trace.mean((2, 3), keepdim=True) if mean_support is None else masked_mean(trace, mean_support)
            mass = m.norm(dim=2) * (trace > tr_rel * ref) * support
            captured["agreement"] = weight.sum((-2, -1)) / mass.sum((-2, -1)).clamp_min(1e-6)
        return weight

    lock_nodes.coherence = diagnostic
    try:
        yield captured
    finally:
        lock_nodes.coherence = original


class NodeTracker:
    def __init__(self, batch, sigma, *, static_residual=True, support_mask=None,
                 return_diagnostics=False):
        self.lattice = named_nodes("v17")
        self.rho = torch.zeros(batch, 17, 2)
        self.evidence = torch.zeros(batch, 17)
        self.locked = torch.zeros(batch, 17, dtype=torch.bool)
        self.perms = rotation_perms(self.lattice)
        self.sigma = sigma
        self.static_residual = static_residual
        self.support_mask = cone_mask()[7:26, 16:33] if support_mask is None else support_mask
        self.return_diagnostics = return_diagnostics

    @property
    def nu(self):
        return self.lattice[None] + self.rho

    def rotate(self, index, action):
        k = turn_k(action)
        source = self.perms[(-k) % 4]
        self.rho[index] = rot_vec(self.rho[index, source], k)
        self.evidence[index] = self.evidence[index, source]
        self.locked[index] = self.locked[index, source]

    @torch.inference_mode()
    def step(self, previous, current, actions):
        turns = (actions == 0) | (actions == 1)
        for i in turns.nonzero().flatten().tolist():
            self.rotate(i, actions[i])
        measured = self.nu.clone()
        evidence = torch.zeros_like(self.evidence)
        agreement = torch.zeros_like(self.evidence)
        before = torch.zeros_like(self.evidence)
        prior = self.nu.clone()
        ids = (~turns).nonzero().flatten()
        if len(ids):
            nu = self.nu[ids]
            with agreement_diagnostic() as diagnostic:
                output = lock_nodes.measure_nodes(
                    previous[ids], current[ids], nu, self.lattice, camera_action(actions[ids]),
                    world=33, dt=1, box=BOX, support_mask=self.support_mask,
                    sigma=self.sigma, radius=None, static_residual=self.static_residual,
                    return_diagnostics=self.return_diagnostics)
                value, ev, locked = output[:3]
                if self.return_diagnostics:
                    before[ids] = output[3]["evidence_before_modes"]
            ev[:, 0] = 0
            cumulative = self.evidence[ids] + ev
            gain = ev[..., None] / cumulative[..., None].clamp_min(1e-6)
            # Same fp32 residual, per-component step clip and rho bound as _forward_lock.
            step = (gain * (value - nu)).clamp(-0.3, 0.3)
            self.rho[ids] = (nu - self.lattice[None] + step).clamp(-0.75, 0.75)
            self.evidence[ids] = cumulative
            self.locked[ids] = locked.bool()
            measured[ids], evidence[ids] = value, ev
            agreement[ids] = diagnostic["agreement"]
        self.rho[:, 0] = 0
        self.evidence[:, 0] = 0
        self.locked[:, 0] = False
        measured[:, 0] = 0
        agreement[:, 0] = 0
        result = dict(measured=measured, velocity=self.nu.clone(), evidence=evidence,
                    agreement=agreement, locked=self.locked.clone(), turns=turns,
                    active=self.locked & (self.evidence > EVIDENCE_FLOOR))
        if self.return_diagnostics:
            result.update(evidence_before_modes=before, prior=prior)
        return result
