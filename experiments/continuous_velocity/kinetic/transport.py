"""Spectral periodic shifts and sub-pixel refinement of lattice correlation peaks.

``spectral_shift`` is adapted from kinetic-world-models (``kwm/transport.py``,
Copyright (c) denzelzheng, MIT License). Differences from that file: Nyquist bins
use the symmetric cosine convention so that the interpolant is real and even, and
the broadcasting rule is documented below. A single shift is used per operation;
no claim is made about exact composition of many fractional shifts, which fails
on the Nyquist bins of an even-sized grid.

Conventions (identical to the parent model): maps are ``[..., H, W]`` with rows
as ``y`` and columns as ``x``; a shift ``(dx, dy)`` moves content at ``(x, y)``
to ``(x+dx, y+dy)`` periodically, so an integer shift equals
``torch.roll(h, shifts=(dy, dx), dims=(-2, -1))``.
"""
import math

import torch

# Denominator floor of the normalized similarity. Below it the clamped quotient's
# derivatives are undefined, so any refinement whose start or end point sits at
# the floor is rejected (an audit of this code found the earlier ``d > 0`` guard
# let such steps through).
DENOMINATOR_FLOOR = 1e-5


def _axis_phase(n, delta, deriv=0, half=False):
    """Phase basis e^{-2*pi*i*k*delta} per frequency of one axis, with Nyquist as cos.

    ``delta`` has any shape; the result appends one frequency dimension. With
    ``deriv`` > 0 the derivative with respect to ``delta`` is returned (used by
    the peak interpolant, which uses the +i convention; see ``_interpolant``).
    """
    device, dtype = delta.device, torch.float32
    k = (torch.fft.rfftfreq(n, device=device, dtype=dtype) if half
         else torch.fft.fftfreq(n, device=device, dtype=dtype))
    nyquist = k.abs() == .5
    omega = 2*math.pi*k
    angle = omega*delta[..., None]
    # complex bins: d^m/d delta^m e^{+i omega delta} = (i omega)^m e^{i omega delta}
    basis = torch.polar(torch.ones_like(angle), angle)*(1j*omega)**deriv
    # Nyquist bins: cos(pi delta) and its derivatives, real valued.
    cos_terms = [torch.cos, lambda a: -torch.sin(a), lambda a: -torch.cos(a)]
    nyq = cos_terms[deriv](math.pi*delta[..., None])*math.pi**deriv
    return torch.where(nyquist, nyq.to(basis.dtype), basis)


def spectral_shift(fields, shift):
    """Shift real periodic maps ``[..., H, W]`` by real displacements ``[..., 2]``.

    ``shift`` leading dimensions are aligned with the *leading* dimensions of
    ``fields`` (a ``[B, 2]`` shift applies to ``[B, T, C, H, W]`` fields by
    broadcasting over ``T`` and ``C``). Differentiable in both arguments.
    """
    h, w = fields.shape[-2:]
    lead = fields.dim()-2
    s = shift
    while s.dim()-1 < lead:
        s = s.unsqueeze(-2)
    # Content moves by +d, so the spectrum is multiplied by e^{-i omega d}.
    phase_x = _axis_phase(w, -s[..., 0], half=True)          # [..., W//2+1]
    phase_y = _axis_phase(h, -s[..., 1])                      # [..., H]
    phase = phase_y[..., :, None]*phase_x[..., None, :]
    spectrum = torch.fft.rfft2(fields.float())
    out = torch.fft.irfft2(spectrum*phase, s=(h, w))
    return out.to(fields.dtype)


def raised_cosine_taper(size=32, width=3, device='cpu'):
    """Separable window equal to 1 in the interior and falling to 0 at the border.

    ``width`` pixels at each edge ramp as a raised cosine. Applied to both field
    and mask before scattering, so a spectral shift of the windowed observation
    rings far less than a shift of the hard camera window would.
    """
    ramp = torch.arange(size, device=device, dtype=torch.float32)
    edge = torch.minimum(ramp, size-1-ramp)
    taper = .5-.5*torch.cos(math.pi*(edge+.5)/width)
    taper = torch.where(edge+.5 >= width, torch.ones_like(taper), taper)
    return taper[:, None]*taper[None, :]


def _interpolant(spectrum, delta, hermitian_weight, deriv_x=0, deriv_y=0):
    """Band-limited interpolant of ``irfft2(spectrum)`` and its derivatives.

    ``spectrum`` ``[..., H, W//2+1]`` is an rfft2 of a real lattice function
    ``f``; ``delta`` ``[..., 2]`` is a real position. Returns the chosen partial
    derivative of ``f(delta) = (1/HW) sum_k c_k Re[F_k e^{+i omega_k . delta}]``
    with Hermitian doubling weights ``c_k`` and the cosine convention on
    Nyquist bins. At integer ``delta`` this equals the irfft2 values.
    """
    h = spectrum.shape[-2]
    w = 2*(spectrum.shape[-1]-1)
    phase_x = _axis_phase(w, delta[..., 0], deriv_x, half=True)*hermitian_weight
    phase_y = _axis_phase(h, delta[..., 1], deriv_y)
    value = torch.einsum('...hw,...h,...w->...', spectrum, phase_y, phase_x)
    return value.real/(h*w)


def _hermitian_weights(w, device):
    weight = torch.full((w//2+1,), 2., device=device)
    weight[0] = 1.
    if w % 2 == 0:
        weight[-1] = 1.
    return weight


def refine_peak(numerator_spectrum, denominator_spectrum, start, steps=3,
                max_step=.5, radius=1., eigenvalue_margin=1e-3):
    """Newton refinement of the maximum of S = 2 N / D on the trigonometric interpolant.

    ``numerator_spectrum``/``denominator_spectrum`` are rfft2 spectra of the
    lattice functions N and D (``[..., H, W//2+1]``); ``start`` ``[..., 2]`` is
    the initial continuous displacement (the attention expectation). The Hessian
    is detached and used as a preconditioner; gradients flow through the
    Newton residual, the spectra and ``start``. The refinement is accepted only
    if the final Hessian is negative definite with a relative margin, the point
    moved by at most ``radius`` pixels, the denominator kept at least half of its
    initial value and stayed above ``DENOMINATOR_FLOOR`` at both the start and
    the end, and the similarity improved without exceeding 1 (+1e-3).
    Otherwise the start value is returned. Returns ``(refined, accepted)``.
    """
    w = 2*(numerator_spectrum.shape[-1]-1)
    weights = _hermitian_weights(w, start.device)
    if w != 50 or numerator_spectrum.shape[-2] != 50:
        raise ValueError('Peak refinement assumes the 50x50 world grid (even width)')

    h = numerator_spectrum.shape[-2]
    spectra = torch.stack([numerator_spectrum, denominator_spectrum], 0)       # [2, ..., H, W//2+1]

    def evaluate(delta):
        # Shared contractions: three row contractions (f, d/dy, d2/dy2), then short column ones.
        phase_y = [_axis_phase(h, delta[..., 1], m) for m in range(3)]
        phase_x = [_axis_phase(w, delta[..., 0], n, half=True)*weights for n in range(3)]
        rows = [torch.einsum('s...hw,...h->s...w', spectra, phase_y[m]) for m in range(3)]

        def value(m, n):
            return torch.einsum('s...w,...w->s...', rows[m], phase_x[n]).real/(h*w)
        f, fx, fy = value(0, 0), value(0, 1), value(1, 0)
        fxx, fxy, fyy = value(0, 2), value(1, 1), value(2, 0)
        n, d = f[0], f[1].clamp_min(DENOMINATOR_FLOOR)
        nx, ny, dx, dy = fx[0], fy[0], fx[1], fy[1]
        s = 2*n/d
        grad = torch.stack([2*(nx*d-n*dx)/d**2, 2*(ny*d-n*dy)/d**2], -1)

        def second(nab, na, nb, dab, da, db):
            return 2*(nab/d-(na*db+nb*da)/d**2-n*dab/d**2+2*n*da*db/d**3)
        sxx = second(fxx[0], nx, nx, fxx[1], dx, dx)
        sxy = second(fxy[0], nx, ny, fxy[1], dx, dy)
        syy = second(fyy[0], ny, ny, fyy[1], dy, dy)
        hess = torch.stack([torch.stack([sxx, sxy], -1), torch.stack([sxy, syy], -1)], -2)
        return s, grad, hess, f[1]

    s0, _, _, d0 = evaluate(start)
    delta = start
    for _ in range(steps):
        _, grad, hess, _ = evaluate(delta)
        hd = hess.detach()
        det = hd[..., 0, 0]*hd[..., 1, 1]-hd[..., 0, 1]**2
        # Newton step -H^{-1} g with a closed-form inverse; (nearly) singular H
        # relative to its own scale gives no step.
        inverse = torch.stack([torch.stack([hd[..., 1, 1], -hd[..., 0, 1]], -1),
                               torch.stack([-hd[..., 0, 1], hd[..., 0, 0]], -1)], -2)
        scale = (hd[..., 0, 0]+hd[..., 1, 1]).square()+1e-12
        safe = det.abs() > 1e-6*scale
        inverse = inverse/torch.where(safe, det, torch.ones_like(det))[..., None, None]
        step = -(inverse@grad[..., None])[..., 0]
        step = torch.where(safe[..., None], step, torch.zeros_like(step)).clamp(-max_step, max_step)
        delta = delta+step
    s1, _, hess, d1 = evaluate(delta)
    hd = hess.detach()
    trace = hd[..., 0, 0]+hd[..., 1, 1]
    det = hd[..., 0, 0]*hd[..., 1, 1]-hd[..., 0, 1]**2
    # Negative definite with margin: both eigenvalues below -margin*|trace|/2.
    negative = (trace < 0) & (det > 0)
    largest = trace/2+torch.sqrt((trace/2).square()-det).nan_to_num(0.)
    negative = negative & (largest < -eigenvalue_margin*trace.abs()/2)
    moved = (delta-start).norm(dim=-1) <= radius
    positive = (d0.detach() > DENOMINATOR_FLOOR) & (d1.detach() > DENOMINATOR_FLOOR)
    accepted = (negative & moved & positive & (d1.detach() >= .5*d0.detach())
                & (s1.detach() <= 1+1e-3) & (s1.detach() >= s0.detach()))
    refined = torch.where(accepted[..., None], delta, start)
    return refined, accepted
