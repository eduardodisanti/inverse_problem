import numpy as np
from dataclasses import dataclass


# ============================================================
# Bearing configuration
# ============================================================

@dataclass
class BearingParams:
    fs: int
    duration: float
    shaft_freq: float

    noise_std: float = 0.03
    harmonic_decay: float = 0.55

    # structural resonance
    resonance_freq: float = 2500.0
    resonance_decay: float = 0.15


# ------------------------------------------------------------
# Domain configurations
# ------------------------------------------------------------

CWRU_PARAMS = BearingParams(
    fs=12000,
    duration=0.1,
    shaft_freq=30.0,
    noise_std=0.03,
    harmonic_decay=0.55,
    resonance_freq=3000,
    resonance_decay=0.15,
)


NASA_PARAMS = BearingParams(
    fs=20000,
    duration=1.0,          
    shaft_freq=33.33,      
    noise_std=0.04,
    harmonic_decay=0.60,
    resonance_freq=4000,
    resonance_decay=0.20,
)


# ============================================================
# Simulator
# ============================================================

class BearingSignalSimulator:

    def __init__(self, params):
        self.params = params
        self.t = np.arange(
            0,
            params.duration,
            1 / params.fs
        )


    # --------------------------------------------------------
    # Nominal rotating structure
    # --------------------------------------------------------

    def _healthy_base(
        self,
        amp=1.0,
        phase=0.0,
        speed_scale=1.0,
        asset_signature=None,
    ):

        f0 = self.params.shaft_freq * speed_scale

        if asset_signature is None:
            asset_signature = np.ones(5)

        x = np.zeros_like(self.t)

        for k in range(1, 6):

            harmonic_amp = (
                amp
                * self.params.harmonic_decay**(k-1)
                * asset_signature[k-1]
            )

            x += (
                harmonic_amp
                * np.sin(
                    2*np.pi*k*f0*self.t
                    + phase/k
                )
            )


        # structural resonance component
        resonance_amp = 0.05 * np.mean(asset_signature)

        x += (
            resonance_amp
            *
            np.sin(
                2*np.pi*self.params.resonance_freq*self.t
            )
            *
            np.exp(
                -self.params.resonance_decay*self.t
            )
        )

        return x



    # --------------------------------------------------------
    # Fault impulse model
    # --------------------------------------------------------

    def _impulse_train(
        self,
        fault_freq,
        strength=0.5,
        jitter=0.0,
    ):

        period = 1.0 / fault_freq

        pulse_times = (
            np.arange(
                int(self.params.duration / period) + 2
            )
            *
            period
        )


        if jitter > 0:

            pulse_times += np.random.normal(
                0,
                jitter * period,
                size=pulse_times.shape
            )


        pulse_times = pulse_times[
            (pulse_times >= 0)
            &
            (pulse_times < self.params.duration)
        ]


        sig = np.zeros_like(self.t)

        width = max(
            int(0.001*self.params.fs),
            3
        )


        kt = np.linspace(
            -2,
            2,
            width
        )

        kernel = (
            np.exp(-kt**2)
            *
            np.cos(12*kt)
        )

        kernel /= (
            np.max(np.abs(kernel))
            +
            1e-8
        )


        for pt in pulse_times:

            idx = int(
                pt*self.params.fs
            )

            l = max(
                0,
                idx-width//2
            )

            r = min(
                len(sig),
                idx-width//2+width
            )

            sig[l:r] += (
                strength
                *
                kernel[:r-l]
            )

        return sig



    # --------------------------------------------------------
    # Installation nuisance
    # --------------------------------------------------------

    def _nuisance(
        self,
        x,
        offset=0.0,
        extra_noise=0.0,
        dropout=0.0,
    ):

        y = x + offset


        if dropout > 0:

            span = int(
                dropout*len(y)
            )

            if span > 0:

                s = np.random.randint(
                    0,
                    max(
                        1,
                        len(y)-span
                    )
                )

                y[s:s+span] *= np.random.uniform(
                    0.6,
                    1.0
                )


        return (
            y
            +
            np.random.normal(
                0,
                self.params.noise_std + extra_noise,
                len(y)
            )
        )



    # --------------------------------------------------------
    # Signal generation
    # --------------------------------------------------------

    def sample(
        self,
        regime="healthy",
        mc=True,
        coverage="original",
        asset_signature=None,
    ):


        if not mc:

            amp = 1.0
            phase = 0.0
            speed = 1.0
            offset = 0.0
            en = 0.0
            drop = 0.0


        elif coverage == "original":

            amp = np.random.uniform(
                0.85,
                1.15
            )

            phase = np.random.uniform(
                -0.3,
                0.3
            )

            speed = np.random.uniform(
                0.95,
                1.05
            )

            offset = np.random.uniform(
                -0.05,
                0.05
            )

            en = np.random.uniform(
                0,
                0.03
            )

            drop = np.random.uniform(
                0,
                0.03
            )


        elif coverage == "wide":

            amp = np.random.uniform(
                0.75,
                1.25
            )

            phase = np.random.uniform(
                -0.6,
                0.6
            )

            speed = np.random.uniform(
                0.90,
                1.10
            )

            offset = np.random.uniform(
                -0.10,
                0.10
            )

            en = np.random.uniform(
                0,
                0.05
            )

            drop = np.random.uniform(
                0,
                0.05
            )


        else:
            raise ValueError(
                coverage
            )


        base = self._healthy_base(
            amp,
            phase,
            speed,
            asset_signature,
        )


        if regime == "healthy":

            x = base


        elif regime == "outer_race":

            x = (
                base
                +
                self._impulse_train(
                    108*speed,
                    np.random.uniform(
                        0.35,
                        0.70
                    ),
                    0.05,
                )
            )


        elif regime == "inner_race":

            x = (
                base
                +
                self._impulse_train(
                    162*speed,
                    np.random.uniform(
                        0.35,
                        0.80
                    ),
                    0.03,
                )
            )


        elif regime == "ball_fault":

            modulation = (
                1
                +
                0.25
                *
                np.sin(
                    2*np.pi*12*self.t
                )
            )

            x = (
                base
                +
                modulation
                *
                self._impulse_train(
                    72*speed,
                    np.random.uniform(
                        0.30,
                        0.60
                    ),
                    0.06,
                )
            )


        else:

            raise ValueError(
                regime
            )


        return self._nuisance(
            x,
            offset,
            en,
            drop,
        )



    def canonical(self):

        return self.sample(
            regime="healthy",
            mc=False
        )