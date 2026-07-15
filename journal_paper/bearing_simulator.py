import numpy as np
from dataclasses import dataclass

@dataclass
class BearingParams:
    fs: int = 12000
    duration: float = 0.1
    shaft_freq: float = 30.0
    noise_std: float = 0.03
    harmonic_decay: float = 0.55

class BearingSignalSimulator:
    def __init__(self, params=BearingParams()):
        self.params = params
        self.t = np.arange(0, params.duration, 1 / params.fs)

    def _healthy_base(self, amp=1.0, phase=0.0, speed_scale=1.0):
        f0 = self.params.shaft_freq * speed_scale
        x = np.zeros_like(self.t)
        for k in range(1, 6):
            x += (amp * self.params.harmonic_decay**(k-1)) * np.sin(2*np.pi*k*f0*self.t + phase/k)
        return x

    def _impulse_train(self, fault_freq, strength=0.5, jitter=0.0):
        period = 1.0 / fault_freq
        pulse_times = np.arange(int(self.params.duration / period) + 2) * period
        if jitter > 0:
            pulse_times += np.random.normal(0, jitter * period, size=pulse_times.shape)
        pulse_times = pulse_times[(pulse_times >= 0) & (pulse_times < self.params.duration)]
        sig = np.zeros_like(self.t)
        width = max(int(0.001 * self.params.fs), 3)
        kt = np.linspace(-2, 2, width)
        kernel = np.exp(-kt**2) * np.cos(12 * kt)
        kernel /= np.max(np.abs(kernel)) + 1e-8
        for pt in pulse_times:
            idx = int(pt * self.params.fs)
            l, r = max(0, idx - width//2), min(len(sig), idx - width//2 + width)
            sig[l:r] += strength * kernel[:r-l]
        return sig

    def _nuisance(self, x, offset=0.0, extra_noise=0.0, dropout=0.0):
        y = x + offset
        if dropout > 0:
            span = int(dropout * len(y))
            if span > 0:
                s = np.random.randint(0, max(1, len(y) - span))
                y[s:s+span] *= np.random.uniform(0.6, 1.0)
        return y + np.random.normal(0, self.params.noise_std + extra_noise, len(y))

    def canonical(self, regime='healthy'):
        return self.sample(regime=regime, mc=False)

    def sample(self, regime='healthy', mc=True, coverage='original'):
        if not mc:
            amp = 1.0
            phase = 0.0
            speed = 1.0
            offset = 0.0
            en = 0.0
            drop = 0.0

        elif coverage == 'original':
            amp    = np.random.uniform(0.85, 1.15)
            phase  = np.random.uniform(-0.3, 0.3)
            speed  = np.random.uniform(0.95, 1.05)
            offset = np.random.uniform(-0.05, 0.05)
            en     = np.random.uniform(0.0, 0.03)
            drop   = np.random.uniform(0.0, 0.03)

        elif coverage == 'wide':
            amp    = np.random.uniform(0.75, 1.25)
            phase  = np.random.uniform(-0.6, 0.6)
            speed  = np.random.uniform(0.90, 1.10)
            offset = np.random.uniform(-0.10, 0.10)
            en     = np.random.uniform(0.0, 0.05)
            drop   = np.random.uniform(0.0, 0.05)

        else:
            raise ValueError(f"Unknown coverage mode: {coverage}")

        base = self._healthy_base(amp, phase, speed)

        if regime == 'healthy':
            x = base
        elif regime == 'outer_race':
            x = base + self._impulse_train(
                108 * speed,
                np.random.uniform(0.35, 0.70),
                0.05,
            )
        elif regime == 'inner_race':
            x = base + self._impulse_train(
                162 * speed,
                np.random.uniform(0.35, 0.80),
                0.03,
            )
        elif regime == 'ball_fault':
            mod = 1 + 0.25 * np.sin(2 * np.pi * 12 * self.t)
            x = base + mod * self._impulse_train(
                72 * speed,
                np.random.uniform(0.30, 0.60),
                0.06,
            )
        else:
            raise ValueError(regime)

        return self._nuisance(x, offset, en, drop)