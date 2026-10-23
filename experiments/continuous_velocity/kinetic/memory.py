"""Kinetic extension of the continuous-velocity memory.

Three independent changes to ``ContinuousMemory('physical')``, each switchable:

* ``align``/``render`` in {'bilinear', 'spectral'}: context alignment and future
  rendering by periodic bilinear sampling (parent model) or by a Fourier phase
  shift (kinetic-world-models). Spectral alignment tapers the observed window
  with a raised cosine before scattering; the coverage support used for the
  normalized-convolution fusion is the *bilinear*-sampled tapered mask (smooth,
  exactly zero outside the window), and aligned values are zeroed where that
  support is below 0.05, because a phase-shifted mask rings over the whole
  zero-padded region and dividing by it would amplify ringing.
* ``refine`` in {'none', 'newton'}: sub-pixel Newton refinement of every pair
  displacement on the trigonometric interpolant of the masked similarity.
* ``order`` in {1, 2}: constant velocity, or velocity plus acceleration fitted
  jointly by weighted least squares with robust reweighting and an
  evidence-adaptive gate on the acceleration (``kappa``; 0 disables the gate;
  the default 9 is a three-sigma rule after an effective-sample-size
  correction, because the lag-1/2/4 pairs share frames).

With ``align='bilinear', render='bilinear', refine='none', order=1`` the model
is the parent physical model, parameter for parameter and value for value.
Formal inputs are unchanged: context images, camera origins, timestamps,
future camera origins and target times. Overrides exist only for labeled
oracle diagnostics.
"""
import math
from pathlib import Path
import sys

import torch
from torch import nn

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from model import ContinuousMemory, CorrespondenceAttention, fit_velocity, periodic_sample, world_embed  # noqa: E402
from kinetic.transport import raised_cosine_taper, refine_peak, spectral_shift  # noqa: E402

ALIGN = ('bilinear', 'spectral')
RENDER = ('bilinear', 'spectral')
REFINE = ('none', 'newton')
ORDERS = (1, 2)
ROBUST_SCALE = .15
SUPPORT_FLOOR = .05
DEFAULT_KAPPA = 9.


def kinetic_observations(fields, mask, times, shifts, temperature, confidence=None, refine='none'):
    """The parent's masked FFT correspondence, plus pair times and optional refinement.

    Returns the parent's observation dictionary (displacements, dt, weights,
    confidence_features) with ``s`` (the acceleration regressor relative to the
    last context time), ``lattice_displacements`` (the unrefined expectation) and
    ``refined`` (acceptance mask). With ``refine='none'`` the displacements equal
    ``paper.mechanism._observations`` exactly.
    """
    batch, length = fields.shape[:2]
    if length < 5:
        raise ValueError('At least five context frames are required')
    if not torch.all(times[:, 1:] > times[:, :-1]):
        raise ValueError('Context timestamps must be strictly increasing')
    fft = torch.fft.rfft2(fields)
    energy = torch.fft.rfft2(fields.square().sum(2))
    maskfft = torch.fft.rfft2(mask[:, :, 0])
    raw_energy = torch.fft.rfft2(fields[:, :, 0].square())
    first, second = [], []
    for lag in (1, 2, 4):
        starts = torch.linspace(0, length-lag-1, 12, device=fields.device).long()
        first.append(starts)
        second.append(starts+lag)
    i, j = torch.cat(first), torch.cat(second)
    dt = times[:, j]-times[:, i]
    reference = times[:, -1:]
    s = .5*((times[:, j]-reference).square()-(times[:, i]-reference).square())
    numerator_spectrum = (fft[:, i].conj()*fft[:, j]).sum(2)
    denominator_spectrum = energy[:, i].conj()*maskfft[:, j]+maskfft[:, i].conj()*energy[:, j]
    numerator = torch.fft.irfft2(numerator_spectrum, s=(50, 50)).flatten(2)
    denominator = torch.fft.irfft2(denominator_spectrum, s=(50, 50)).flatten(2)
    raw_overlap = torch.fft.irfft2(
        raw_energy[:, i].conj()*maskfft[:, j]+maskfft[:, i].conj()*raw_energy[:, j],
        s=(50, 50)).flatten(2).clamp_min(0)
    similarity = (2*numerator/denominator.clamp_min(1e-5)).clamp(-1, 1)
    allowed = (shifts.abs()[None, None] <= (2.5*dt[:, :, None, None]+1)).all(-1)
    allowed = allowed & (raw_overlap > 2)
    logits = similarity*temperature
    attention = logits.masked_fill(~allowed, -10000).softmax(-1)
    displacement = attention@shifts
    match = (attention*similarity).sum(-1)
    support = (attention*raw_overlap).sum(-1)
    entropy = -(attention*attention.clamp_min(1e-10).log()).sum(-1)/8
    confidence_input = torch.stack([match, entropy, torch.log1p(support)/6], -1)
    reliability = (confidence(confidence_input).squeeze(-1).sigmoid()
                   if confidence is not None else torch.full_like(match, .5))
    weights = reliability*match.clamp_min(0).pow(4)*(support/(support+10))*allowed.any(-1)
    lattice = displacement
    accepted = torch.zeros_like(match, dtype=torch.bool)
    if refine == 'newton':
        displacement, accepted = refine_peak(numerator_spectrum, denominator_spectrum, lattice)
    elif refine != 'none':
        raise ValueError(f'Unknown refinement: {refine}')
    return dict(displacements=displacement, dt=dt, s=s, weights=weights,
                confidence_features=confidence_input, lattice_displacements=lattice,
                refined=accepted)


def kinetic_summary(observations):
    """Temporally resolved observation summary for the learned-offset controls.

    Per lag group (pairs are ordered by start frame): weighted mean velocity of
    the earlier six pairs, of the later six pairs, and the mean weight. The
    parent's summary averages each whole group, whose mean midpoint is the same
    for every lag, so it carries no curvature information; this one does.
    """
    displacement, dt, weights = (observations[k] for k in ('displacements', 'dt', 'weights'))
    parts = []
    for group in range(3):
        for half in range(2):
            sl = slice(group*12+half*6, group*12+half*6+6)
            wg = weights[:, sl]
            parts.append((wg[:, :, None]*displacement[:, sl]/dt[:, sl, None]).sum(1)
                         / wg.sum(1, keepdim=True).clamp_min(1e-6))
    parts.append(torch.stack([weights[:, g*12:(g+1)*12].mean(1) for g in range(3)], -1))
    return torch.cat(parts, -1)


SUMMARY_DIM = 15


def _solve(displacements, dt, s, weights):
    """Joint weighted least squares for (v, a) with d_k = v dt_k + a s_k.

    Falls back to the order-1 solution exactly where the normal matrix is
    (nearly) singular. Returns v, a, the normal-matrix entries and a flag.
    """
    w = weights
    a11 = (w*dt.square()).sum(1, keepdim=True)
    a12 = (w*dt*s).sum(1, keepdim=True)
    a22 = (w*s.square()).sum(1, keepdim=True)
    b1 = (w[..., None]*dt[..., None]*displacements).sum(1)
    b2 = (w[..., None]*s[..., None]*displacements).sum(1)
    det = a11*a22-a12.square()
    well = det > 1e-3*a11*a22
    safe_det = torch.where(well, det, torch.ones_like(det))
    v2 = (a22*b1-a12*b2)/safe_det
    a2 = (a11*b2-a12*b1)/safe_det
    v1 = b1/a11.clamp_min(1e-6)
    v = torch.where(well, v2, v1)
    a = torch.where(well, a2, torch.zeros_like(a2))
    return v, a, dict(a11=a11, a12=a12, a22=a22, b1=b1, well=well)


def fit_motion(displacements, dt, s, weights, order=1, kappa=4.):
    """Robust constant-velocity (order 1) or constant-acceleration (order 2) fit.

    Order 1 reproduces the parent's ``fit_velocity`` with two Cauchy steps.
    Order 2 solves jointly for ``(v, a)``, reweights on the *unshrunk* residuals,
    then gates the acceleration per axis by its own standard error from the
    scale-invariant sandwich estimate, ``g = clip(1 - kappa Var(a)/a^2, 0, 1)``,
    and re-solves ``v`` with the gated acceleration fixed. ``kappa=0`` leaves the
    acceleration ungated. Returns ``(v, a, stats)``; ``v`` is the velocity at
    the last context time.
    """
    if order == 1:
        velocity = fit_velocity(displacements, dt, weights)
        for _ in range(2):
            residual = (displacements/dt[:, :, None]-velocity[:, None]).square().sum(-1)
            robust = weights/(1+residual/ROBUST_SCALE**2)
            velocity = fit_velocity(displacements, dt, robust)
        zero = torch.zeros_like(velocity)
        return velocity, zero, dict(acceleration_raw=zero, acceleration_gate=torch.full_like(zero, float('nan')),
                                    acceleration_se=zero, robust_weights=robust, profiled_velocity=velocity)
    if order != 2:
        raise ValueError(f'Unknown motion order: {order}')
    robust = weights
    v, a, normal = _solve(displacements, dt, s, robust)
    for _ in range(2):
        predicted = dt[:, :, None]*v[:, None]+s[:, :, None]*a[:, None]
        residual = ((displacements-predicted)/dt[:, :, None]).square().sum(-1)
        robust = weights/(1+residual/ROBUST_SCALE**2)
        v, a, normal = _solve(displacements, dt, s, robust)
    # Sandwich variance of a per axis: sigma^2 r^T B r with r = (-A12, A11)/det.
    predicted = dt[:, :, None]*v[:, None]+s[:, :, None]*a[:, None]
    residual = displacements-predicted                                   # [b,k,2] pixels
    total = robust.sum(1, keepdim=True).clamp_min(1e-6)
    sigma2 = (robust[..., None]*residual.square()).sum(1)/total          # [b,2]
    w2 = robust.square()
    b11 = (w2*dt.square()).sum(1, keepdim=True)
    b12 = (w2*dt*s).sum(1, keepdim=True)
    b22 = (w2*s.square()).sum(1, keepdim=True)
    a11, a12, a22 = normal['a11'], normal['a12'], normal['a22']
    det = (a11*a22-a12.square())
    det = torch.where(normal['well'], det, torch.ones_like(det))
    quadratic = (a12.square()*b11-2*a12*a11*b12+a11.square()*b22)/det.square()
    # Effective-sample-size degrees-of-freedom inflation (two fitted parameters).
    n_eff = robust.sum(1, keepdim=True).square()/w2.sum(1, keepdim=True).clamp_min(1e-12)
    inflation = (n_eff/(n_eff-2).clamp_min(1e-3)).clamp_min(1.)
    variance = sigma2*quadratic*inflation                                 # [b,2]
    se = variance.clamp_min(0).sqrt()
    if kappa > 0:
        gate = (1-kappa*variance/(a.square()+1e-12)).clamp(0, 1)
    else:
        gate = torch.ones_like(a)
    gate = torch.where(normal['well'], gate, torch.zeros_like(gate))
    gated = gate*a
    # Profile the velocity with the gated acceleration held fixed; also the
    # constant-velocity profile (a = 0) for the zero-acceleration intervention.
    v = (normal['b1']-a12*gated)/a11.clamp_min(1e-6)
    profiled = normal['b1']/a11.clamp_min(1e-6)
    return v, gated, dict(acceleration_raw=a, acceleration_gate=gate, acceleration_se=se,
                          robust_weights=robust, profiled_velocity=profiled)


class KineticCorrespondence(CorrespondenceAttention):
    """Parent descriptors, temperature and confidence; kinetic observations and fit."""
    def __init__(self, channels=4, refine='none', order=1, kappa=DEFAULT_KAPPA):
        super().__init__(channels)
        if refine not in REFINE or order not in ORDERS:
            raise ValueError((refine, order))
        if not (math.isfinite(kappa) and kappa >= 0):
            raise ValueError('kappa must be finite and nonnegative')
        self.refine, self.order, self.kappa = refine, int(order), float(kappa)

    def observe(self, images, cameras, times, taper=None):
        b, t, h, w = images.shape
        learned = self.encoder(images.reshape(b*t, 1, h, w)).reshape(b, t, -1, h, w)
        descriptors = torch.cat([images[:, :, None], learned], 2)
        fields, mask = world_embed(descriptors, cameras)
        observations = kinetic_observations(fields, mask, times, self.shifts,
                                            self.log_temperature.exp().clamp(5, 150),
                                            self.confidence, self.refine)
        if taper is not None:
            # Matching uses the hard window (as the parent); alignment uses tapered copies.
            fields, mask = world_embed(descriptors*taper, cameras)
            mask = mask*world_embed(taper.expand(b, t, 1, h, w), cameras)[0]
        return fields, mask, observations

    def forward(self, images, cameras, times, taper=None):
        fields, mask, observations = self.observe(images, cameras, times, taper)
        velocity, acceleration, stats = fit_motion(
            observations['displacements'], observations['dt'], observations['s'],
            observations['weights'], self.order, self.kappa)
        details = dict(acceleration=acceleration, **observations, **stats)
        return fields, mask, velocity, kinetic_summary(observations), details


class KineticMemory(ContinuousMemory):
    def __init__(self, align='bilinear', render='bilinear', refine='none', order=1,
                 kappa=DEFAULT_KAPPA, kind='physical', channels=4, taper_width=3):
        super().__init__(kind, channels)
        if align not in ALIGN or render not in RENDER:
            raise ValueError((align, render))
        self.align, self.render, self.order = align, render, int(order)
        # Keep the parent's initialization for the same seed: the parent modules
        # were created in the same order, and the replacement correspondence
        # module copies the parent's freshly initialized weights.
        parent_state = self.correspondence.state_dict()
        self.correspondence = KineticCorrespondence(channels, refine, order, kappa)
        self.correspondence.load_state_dict(parent_state)
        if kind in ('offset', 'relaxed'):
            self.offset_head = nn.Sequential(nn.Linear(SUMMARY_DIM+1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                                             nn.Linear(64, 2))
            if kind == 'relaxed':
                nn.init.zeros_(self.offset_head[-1].weight)
            else:
                nn.init.normal_(self.offset_head[-1].weight, std=.001)
            nn.init.zeros_(self.offset_head[-1].bias)
        self.register_buffer('taper', raised_cosine_taper(32, taper_width), persistent=False)
        self.spec = dict(align=align, render=render, refine=refine, order=int(order),
                         kappa=float(kappa), kind=kind, channels=channels)

    def motion_offsets(self, dt, velocity, acceleration, summary):
        physical = dt[:, :, None]*velocity[:, None]+.5*dt[:, :, None].square()*acceleration[:, None]
        if self.kind == 'physical':
            return physical
        s = summary[:, None].expand(-1, dt.shape[1], -1)
        args = torch.cat([s, dt[:, :, None]/40], -1)
        zero = torch.cat([s, torch.zeros_like(dt[:, :, None])], -1)
        learned = 40*(self.offset_head(args)-self.offset_head(zero))
        return learned+physical if self.kind == 'relaxed' else learned

    def offsets(self, dt, velocity, summary):
        return self.motion_offsets(dt, velocity, torch.zeros_like(velocity), summary)

    def _align(self, fields, mask, offsets):
        b, t = fields.shape[:2]
        if self.align == 'bilinear':
            coords = self.world_grid[None, None]+offsets[:, :, None, None]
            aligned = periodic_sample(fields.reshape(b*t, -1, 50, 50), coords.reshape(b*t, 50, 50, 2)).reshape(b, t, -1, 50, 50)
            visible = periodic_sample(mask.reshape(b*t, 1, 50, 50), coords.reshape(b*t, 50, 50, 2)).reshape(b, t, 1, 50, 50)
            return aligned, visible
        # Reading F(u + offset) is a content shift by -offset. The support is the
        # bilinear-sampled tapered mask: smooth, in [0, 1], exactly zero outside.
        shift = -offsets.reshape(b*t, 2)
        aligned = spectral_shift(fields.reshape(b*t, -1, 50, 50), shift).reshape(b, t, -1, 50, 50)
        coords = self.world_grid[None, None]+offsets[:, :, None, None]
        visible = periodic_sample(mask.reshape(b*t, 1, 50, 50), coords.reshape(b*t, 50, 50, 2)).reshape(b, t, 1, 50, 50)
        supported = (visible >= SUPPORT_FLOOR).to(aligned.dtype)
        return aligned*supported, visible*supported

    def _render(self, canonical, target_cameras, future_offsets):
        b, n = future_offsets.shape[:2]
        if self.render == 'bilinear':
            q = self.view_grid[None, None]+target_cameras[:, :, None, None]-future_offsets[:, :, None, None]
            return periodic_sample(canonical[:, None].expand(-1, n, -1, -1, -1).reshape(b*n, 1, 50, 50),
                                   q.reshape(b*n, 32, 32, 2)).reshape(b, n, 32, 32)
        # View pixel x reads C(x + c - offset): shift content by offset - c, crop the window.
        shift = (future_offsets-target_cameras).reshape(b*n, 2)
        shifted = spectral_shift(canonical[:, None].expand(-1, n, -1, -1, -1).reshape(b*n, 1, 50, 50), shift)
        return shifted[:, 0, :32, :32].reshape(b, n, 32, 32)

    def forward(self, context, cameras, context_times, target_cameras, target_times,
                velocity_override=None, acceleration_override=None, zero_acceleration=None):
        """``zero_acceleration``: None/False, 'both' (constant-velocity profile of the
        same robust fit for alignment and transport) or 'transport' (keep the
        second-order alignment and the velocity at t0, zero only future curvature)."""
        b, t, h, w = context.shape
        taper = self.taper if self.align == 'spectral' else None
        fields, mask, velocity, summary, details = self.correspondence(context, cameras, context_times, taper)
        acceleration = details['acceleration']
        estimated_velocity, estimated_acceleration = velocity, acceleration
        if velocity_override is not None:
            if self.kind != 'physical':
                raise ValueError('Oracle override only defined for physical transport')
            velocity = velocity_override
        if acceleration_override is not None:
            if self.kind != 'physical':
                raise ValueError('Oracle override only defined for physical transport')
            acceleration = acceleration_override
        if zero_acceleration not in (None, False, 'both', 'transport'):
            raise ValueError(f'Unknown zero_acceleration mode: {zero_acceleration}')
        if zero_acceleration == 'both':
            velocity = details['profiled_velocity'] if velocity_override is None else velocity
            acceleration = torch.zeros_like(acceleration)
        dt = context_times-context_times[:, -1:]
        offsets = self.motion_offsets(dt, velocity, acceleration, summary)
        if zero_acceleration == 'transport':
            acceleration = torch.zeros_like(acceleration)
        aligned, visible = self._align(fields, mask, offsets)
        values = aligned/visible.clamp_min(1e-4)
        scores = self.memory_attention(values.reshape(b*t, -1, 50, 50)).reshape(b, t, 1, 50, 50)
        weights = scores.clamp(-8, 8).exp()*visible
        memory = (weights*values).sum(1)/weights.sum(1).clamp_min(1e-5)
        canonical = (memory[:, :1]+self.decoder(memory)).clamp(0, 1)
        future_dt = target_times-context_times[:, -1:]
        future_offsets = self.motion_offsets(future_dt, velocity, acceleration, summary)
        prediction = self._render(canonical, target_cameras, future_offsets)
        auxiliary = dict(velocity=velocity, acceleration=acceleration, offsets=future_offsets,
                         memory=canonical, summary=summary, estimated_velocity=estimated_velocity,
                         estimated_acceleration=estimated_acceleration,
                         oracle_velocity_used=velocity_override is not None,
                         oracle_acceleration_used=acceleration_override is not None,
                         zero_acceleration=zero_acceleration or '')
        for key in ('displacements', 'dt', 's', 'weights', 'lattice_displacements', 'refined',
                    'acceleration_raw', 'acceleration_gate', 'acceleration_se', 'robust_weights',
                    'profiled_velocity'):
            auxiliary[key] = details[key]
        return prediction, auxiliary


def build_kinetic_model(spec):
    """Factory from a plain specification dictionary (stored in run manifests)."""
    spec = dict(spec)
    return KineticMemory(align=spec.get('align', 'bilinear'), render=spec.get('render', 'bilinear'),
                         refine=spec.get('refine', 'none'), order=int(spec.get('order', 1)),
                         kappa=float(spec.get('kappa', DEFAULT_KAPPA)), kind=spec.get('kind', 'physical'),
                         channels=int(spec.get('channels', 4)))
