"""Rendering and observation-only tracking for the frozen velocity experiment."""
import numpy as np

WORLD = 50
WINDOW = 32


def translate(image, xy, interpolate=False):
    """Periodic translation; interpolation always starts from the original image."""
    x, y = xy
    if not interpolate:
        return np.roll(image, (int(np.floor(y + .5)), int(np.floor(x + .5))), (0, 1))
    ix, iy = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - ix, y - iy
    return sum(w * np.roll(image, (iy + dy, ix + dx), (0, 1))
               for dx, dy, w in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                                 (0, 1, (1-fx)*fy), (1, 1, fx*fy)])


def render(template, positions, cameras, interpolate=False):
    frames, visibility = [], []
    for pos, cam in zip(positions, cameras):
        world = translate(template, pos, interpolate)
        view = np.roll(world, tuple((-cam[::-1]).astype(int)), (0, 1))[:WINDOW, :WINDOW]
        frames.append(view)
        visibility.append(float((view**2).sum() / max((world**2).sum(), 1e-8)))
    return np.asarray(frames, dtype=np.float32), np.asarray(visibility)


class Detector:
    """Least-squares template localization on the torus, without true positions.

    Scores all translations, accounting for the part visible through the camera.
    A second template is fitted to the positive residual for ghost diagnostics.
    The template is supplied as an oracle identity to BOTH model and tracker;
    neither localizer receives true position or velocity.
    """
    def __init__(self, template):
        self.template = template
        self.ft = np.fft.fft2(template)
        mask = np.zeros((WORLD, WORLD)); mask[:WINDOW, :WINDOW] = 1
        self.energy = np.fft.ifft2(np.fft.fft2(mask) * np.conj(np.fft.fft2(template**2))).real
        self.total = float((template**2).sum())
        self.eligible = self.energy >= .35 * self.total

    def _fit(self, frame, excluded=None):
        obs = np.zeros((WORLD, WORLD)); obs[:WINDOW, :WINDOW] = np.clip(frame, 0, None)
        dot = np.fft.ifft2(np.fft.fft2(obs) * np.conj(self.ft)).real
        gain = dot**2 / np.maximum(self.energy, 1e-8)
        gain[~self.eligible] = -1
        if excluded is not None:
            yy, xx = np.indices(gain.shape)
            dx = (xx-excluded[0]+25)%50-25; dy = (yy-excluded[1]+25)%50-25
            gain[dx*dx+dy*dy < 8**2] = -1
        y, x = np.unravel_index(gain.argmax(), gain.shape)
        amp = dot[y, x] / max(self.energy[y, x], 1e-8)
        explained = gain[y, x] / max(float((obs**2).sum()), 1e-8)
        present = bool(amp >= .25 and gain[y, x] >= .10*self.total and explained >= .30)
        return np.array([x, y], dtype=float), float(amp), float(explained), present

    def __call__(self, frame):
        pos, amp, quality, present = self._fit(frame)
        fitted = translate(self.template, pos)[:WINDOW, :WINDOW] * amp
        residual = np.clip(frame-fitted, 0, None)
        pos2, amp2, quality2, ghost = self._fit(residual, pos)
        ghost = bool(present and ghost and amp2 >= .35)
        return dict(pos=pos if present else np.array([np.nan, np.nan]),
                    amplitude=amp, quality=quality, present=present, ghost=ghost,
                    second_position=pos2, second_amplitude=amp2,
                    mass=float(np.clip(frame, 0, None).sum()))


def unwrap_track(wrapped, anchor=None, anchor_time=None, speed_bound=2.5):
    """Unwrap using continuity, flag irrecoverable winding ambiguity.

    No GT or true velocity is used. After a gap permitting a half-world traversal,
    or an implausible jump, subsequent absolute winding is treated as unknown.
    """
    result = np.full_like(wrapped, np.nan, dtype=float)
    prev = None if anchor is None else np.asarray(anchor, dtype=float)
    prev_t = anchor_time
    ambiguous = False
    for t, p in enumerate(wrapped):
        if not np.isfinite(p).all() or ambiguous:
            continue
        if prev is None:
            q = p.copy()
        else:
            gap = t-prev_t
            delta = (p-prev+WORLD/2)%WORLD-WORLD/2
            if speed_bound*gap >= WORLD/2 or np.any(np.abs(delta) > speed_bound*gap+1):
                ambiguous = True
                continue
            q = prev+delta
        result[t] = q
        prev, prev_t = q, t
    return result


def fit_velocity(track, min_points=5, min_span=10):
    valid = np.isfinite(track).all(1)
    ts = np.flatnonzero(valid)
    if len(ts) < min_points or np.ptp(ts) < min_span:
        return np.full(2, np.nan), np.full(2, np.nan)
    # Robust median of pairwise slopes with at least ten-frame separation.
    slopes = [(track[j]-track[i])/(j-i) for i in ts for j in ts if j-i >= min_span]
    v = np.median(slopes, axis=0)
    intercept = np.median(track[valid]-ts[:, None]*v, axis=0)
    return v, intercept


def periodic_fit(wrapped):
    """Robust continuous constant-velocity fit to observed positions modulo 50.

    Used to infer winding across occlusions, without GT velocity. Search is over
    continuous velocities on a 0.01 px/frame numerical mesh, then least-squares
    refinement. Reject poorly identified or competing solutions.
    """
    valid = np.isfinite(wrapped).all(1)
    ts = np.flatnonzero(valid)
    empty = (np.full_like(wrapped, np.nan), np.full(2, np.nan), np.full(2, np.nan))
    if len(ts) < 5 or np.ptp(ts) < 10: return empty
    velocities = np.arange(-2.5, 2.5001, .01)
    v, intercept = [], []
    for axis in range(2):
        phases = 2*np.pi/50*(wrapped[valid, axis][None, :]-velocities[:, None]*ts)
        z = np.exp(1j*phases).mean(1)
        score = np.abs(z)
        k = score.argmax(); speed = velocities[k]
        # An alias must be nearly as good and at least .3 px/frame away.
        other = score[np.abs(velocities-speed) >= .3].max()
        if score[k] < .90 or other >= score[k]-.01: return empty
        origin = np.angle(z[k])*50/(2*np.pi)
        q = wrapped[valid, axis]+50*np.round((origin+speed*ts-wrapped[valid, axis])/50)
        for _ in range(3):
            residual = q-(origin+speed*ts)
            keep = np.abs(residual) <= 3
            if keep.sum() < 5: return empty
            speed, origin = np.polyfit(ts[keep], q[keep], 1)
        v.append(speed); intercept.append(origin)
    v, intercept = np.asarray(v), np.asarray(intercept)
    line = intercept+np.arange(len(wrapped))[:, None]*v
    unwrapped = wrapped+50*np.round((line-wrapped)/50)
    unwrapped[np.linalg.norm(unwrapped-line,axis=1)>4] = np.nan
    return unwrapped, v, intercept
