"""Mechanism controls for continuous-motion memory.

The unchanged pilot remains the reference implementation. ``build_model`` has
the same forward signature as its ContinuousMemory, and returns the same core
auxiliary fields. All interventions happen before memory construction, so the
chosen velocity controls *both* context alignment and future transport.
"""
from copy import deepcopy
import math

import torch
from torch import nn

try:
    from ..model import (ContinuousMemory, CorrespondenceAttention, fit_velocity,
                         periodic_sample, world_embed)
except ImportError:  # direct scripts with continuous_velocity on sys.path
    from model import (ContinuousMemory, CorrespondenceAttention, fit_velocity,
                       periodic_sample, world_embed)


def _observations(fields, mask, times, shifts, temperature, confidence=None):
    """The pilot's masked FFT correspondence calculation, before velocity fit.

    ``fields`` may contain raw pixels alone or raw + learned descriptors. This
    is a displacement search on a pixel lattice, not a velocity-channel bank:
    the soft attention expectation yields continuous displacements.
    """
    batch, length = fields.shape[:2]
    if length < 5:
        raise ValueError("At least five context frames are required")
    if not torch.all(times[:, 1:] > times[:, :-1]):
        raise ValueError("Context timestamps must be strictly increasing")
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
    numerator = torch.fft.irfft2(
        (fft[:, i].conj()*fft[:, j]).sum(2), s=(50, 50)).flatten(2)
    denominator = torch.fft.irfft2(
        energy[:, i].conj()*maskfft[:, j]+maskfft[:, i].conj()*energy[:, j],
        s=(50, 50)).flatten(2)
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
    return dict(displacements=displacement, dt=dt, weights=weights,
                confidence_features=confidence_input)


def _physical_fit(observations):
    displacement, dt, weights = (observations[k] for k in ('displacements', 'dt', 'weights'))
    velocity = fit_velocity(displacement, dt, weights)
    for _ in range(2):
        residual = (displacement/dt[:, :, None]-velocity[:, None]).square().sum(-1)
        robust = weights/(1+residual/.15**2)
        velocity = fit_velocity(displacement, dt, robust)
    return velocity


def _summary(observations):
    displacement, dt, weights = (observations[k] for k in ('displacements', 'dt', 'weights'))
    summaries = []
    for group in range(3):
        sl = slice(group*12, (group+1)*12)
        wg = weights[:, sl]
        summaries.append((wg[:, :, None]*displacement[:, sl]/dt[:, sl, None]).sum(1)
                         / wg.sum(1, keepdim=True).clamp_min(1e-6))
    return torch.cat(summaries + [torch.stack([
        weights[:, group*12:(group+1)*12].mean(1) for group in range(3)], -1)], -1)


class PairwiseVelocityMLP(nn.Module):
    """Direct learned regression from the same correspondence observations.

    No fitted velocity, displacement/time ratio, or velocity labels are passed
    to this head. It must learn the relation between displacement and elapsed
    time. Pooling is invariant to pair ordering; confidence gates the evidence.
    """
    def __init__(self):
        super().__init__()
        self.pair_encoder = nn.Sequential(nn.Linear(7, 64), nn.SiLU(),
                                          nn.Linear(64, 64), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.normal_(self.readout[-1].weight, std=.001)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, observations):
        weights = observations['weights']
        features = torch.cat([observations['displacements']/10,
                              observations['dt'][:, :, None]/10,
                              weights[:, :, None], observations['confidence_features']], -1)
        encoded = self.pair_encoder(features)
        pooled = (encoded*weights[:, :, None]).sum(1)/weights.sum(1, keepdim=True).clamp_min(1e-6)
        return self.readout(pooled)


class MechanismCorrespondence(CorrespondenceAttention):
    def __init__(self, channels=4, matching='learned', estimator='fit'):
        super().__init__(channels)
        if matching not in ('learned', 'raw') or estimator not in ('fit', 'mlp'):
            raise ValueError((matching, estimator))
        self.matching, self.estimator = matching, estimator
        if estimator == 'mlp':
            self.velocity_head = PairwiseVelocityMLP()
        if matching == 'raw':
            # Encoder still supplies the identical learned memory features.
            # Matching temperature/confidence stay fixed in this control.
            self.log_temperature.requires_grad_(False)
            self.confidence.requires_grad_(False)

    def forward(self, images, cameras, times):
        b, t, h, w = images.shape
        learned = self.encoder(images.reshape(b*t, 1, h, w)).reshape(b, t, -1, h, w)
        fields, mask = world_embed(torch.cat([images[:, :, None], learned], 2), cameras)
        raw = self.matching == 'raw'
        observations = _observations(
            fields[:, :, :1] if raw else fields, mask, times, self.shifts,
            fields.new_tensor(math.exp(3.7)) if raw else self.log_temperature.exp().clamp(5, 150),
            None if raw else self.confidence)
        velocity = (self.velocity_head(observations) if self.estimator == 'mlp'
                    else _physical_fit(observations))
        return fields, mask, velocity, _summary(observations), observations


class RawPixelCorrespondence(nn.Module):
    """Fully nonlearned, template-free raw-pixel matching and velocity fit."""
    def __init__(self):
        super().__init__()
        yy, xx = torch.meshgrid(torch.arange(50), torch.arange(50), indexing='ij')
        self.register_buffer('shifts', torch.stack([(xx+25)%50-25, (yy+25)%50-25], -1).reshape(-1, 2).float())

    def forward(self, images, cameras, times):
        fields, mask = world_embed(images[:, :, None], cameras)
        observations = _observations(fields, mask, times, self.shifts,
                                     fields.new_tensor(math.exp(3.7)))
        return fields, mask, _physical_fit(observations), _summary(observations), observations


class ZeroScores(nn.Module):
    """Zero logits produce exactly visibility-weighted mean fusion."""
    def forward(self, values):
        return torch.zeros_like(values[:, :1])


def build_model(kind='physical', channels=4):
    """Factory compatible with the original training loop's model interface.

    raw_correlation: nonlearned raw matching; trainable features, fusion, decoder.
    raw_untrained: no trainable parameters anywhere, no supplied digit template.
    mlp_velocity: learned correspondence + direct MLP instead of analytic fitting.
    mean_memory: original correspondence/decoder, visibility-weighted mean fusion.
    """
    if kind in ('physical', 'offset', 'relaxed'):
        return ContinuousMemory(kind, channels)
    model = ContinuousMemory('physical', channels)
    if kind in ('raw_correlation', 'mlp_velocity'):
        original_state = model.correspondence.state_dict()
        model.correspondence = MechanismCorrespondence(
            channels, matching='raw' if kind == 'raw_correlation' else 'learned',
            estimator='mlp' if kind == 'mlp_velocity' else 'fit')
        model.correspondence.load_state_dict(original_state, strict=kind != 'mlp_velocity')
    elif kind == 'mean_memory':
        model.memory_attention = ZeroScores()
    elif kind == 'raw_untrained':
        model.correspondence = RawPixelCorrespondence()
        model.memory_attention = ZeroScores()
        model.decoder = ZeroScores()
    else:
        raise ValueError(f'Unknown mechanism model: {kind}')
    model.mechanism_kind = kind
    return model


def mean_memory_copy(model):
    """Inference intervention retaining the exact fitted encoder and decoder."""
    result = deepcopy(model)
    result.memory_attention = ZeroScores()
    result.mechanism_kind = 'mean_memory_intervention'
    return result


def raw_matching_copy(model):
    """Replace learned matching only; retain fitted memory features/renderer."""
    if model.kind != 'physical':
        raise ValueError('Raw matching intervention requires physical transport')
    result = deepcopy(model)
    channels = model.correspondence.encoder[-1].out_channels
    replacement = MechanismCorrespondence(channels, matching='raw')
    replacement.load_state_dict(model.correspondence.state_dict(), strict=True)
    reference = next(model.parameters())
    replacement.to(device=reference.device, dtype=reference.dtype)
    replacement.train(model.training)
    result.correspondence = replacement
    result.mechanism_kind = 'raw_matching_intervention'
    return result


def _render_velocity(model, fields, mask, velocity, summary, context_times,
                     target_cameras, target_times, details, alignment_velocity=None):
    """Exact pilot alignment/fusion/rendering with one explicitly chosen velocity."""
    b, t = fields.shape[:2]
    dt = context_times-context_times[:, -1:]
    offsets = model.offsets(dt, velocity if alignment_velocity is None else alignment_velocity, summary)
    coords = model.world_grid[None, None]+offsets[:, :, None, None]
    aligned = periodic_sample(fields.reshape(b*t, -1, 50, 50), coords.reshape(b*t, 50, 50, 2)).reshape(b, t, -1, 50, 50)
    visible = periodic_sample(mask.reshape(b*t, 1, 50, 50), coords.reshape(b*t, 50, 50, 2)).reshape(b, t, 1, 50, 50)
    values = aligned/visible.clamp_min(1e-4)
    scores = model.memory_attention(values.reshape(b*t, -1, 50, 50)).reshape(b, t, 1, 50, 50)
    weights = scores.clamp(-8, 8).exp()*visible
    memory = (weights*values).sum(1)/weights.sum(1).clamp_min(1e-5)
    canonical = (memory[:, :1]+model.decoder(memory)).clamp(0, 1)
    future_offsets = model.offsets(target_times-context_times[:, -1:], velocity, summary)
    q = model.view_grid[None, None]+target_cameras[:, :, None, None]-future_offsets[:, :, None, None]
    n = target_times.shape[1]
    prediction = periodic_sample(canonical[:, None].expand(-1, n, -1, -1, -1).reshape(b*n, 1, 50, 50),
                                 q.reshape(b*n, 32, 32, 2)).reshape(b, n, 32, 32)
    return prediction, dict(velocity=velocity, offsets=future_offsets, memory=canonical,
                            summary=summary, **details)


class VelocityIntervention(nn.Module):
    """Inference-time continuous/grid velocity with all model weights fixed.

    Set grid_spacing to None, 1., or .5. Supply ``velocity_override`` only for an
    explicitly labeled oracle diagnostic; omission uses inferred velocity.
    Quantization uses torch.round (ties to even) and has no range clipping.
    scope='both' affects both memory alignment and prediction; scope='transport'
    builds memory with the unquantized selected (estimated/oracle) velocity and
    only quantizes future transport. This is a controlled single
    motion intervention, not an emulation of FloWM's channel-wise max readout.
    """
    def __init__(self, model, grid_spacing=None, scope='both'):
        super().__init__()
        if model.kind != 'physical':
            raise ValueError('Velocity intervention requires physical transport')
        if grid_spacing is not None and (not math.isfinite(grid_spacing) or grid_spacing <= 0):
            raise ValueError('grid_spacing must be finite and positive')
        if scope not in ('both', 'transport'):
            raise ValueError("scope must be 'both' or 'transport'")
        self.model, self.grid_spacing = model, grid_spacing
        self.scope = scope
        self.kind = 'physical'

    def forward(self, context, cameras, context_times, target_cameras, target_times,
                velocity_override=None):
        fields, mask, estimated, summary, details = self.model.correspondence(context, cameras, context_times)
        velocity = estimated if velocity_override is None else velocity_override
        if velocity.shape != estimated.shape:
            raise ValueError('velocity_override must have shape [batch, 2]')
        unquantized = velocity
        if self.grid_spacing is not None:
            velocity = torch.round(velocity/self.grid_spacing)*self.grid_spacing
        alignment_velocity = unquantized if self.scope == 'transport' else velocity
        prediction, auxiliary = _render_velocity(self.model, fields, mask, velocity, summary,
                                                  context_times, target_cameras, target_times, details,
                                                  alignment_velocity=alignment_velocity)
        auxiliary['estimated_velocity'] = estimated
        auxiliary['alignment_velocity'] = alignment_velocity
        auxiliary['oracle_velocity_used'] = velocity_override is not None
        auxiliary['grid_spacing'] = self.grid_spacing
        auxiliary['intervention_scope'] = self.scope
        return prediction, auxiliary
