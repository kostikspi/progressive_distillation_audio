import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is None:
            pass
        else:
            raise NotImplementedError
        return self


wavegrad_params = AttrDict(
    # Training params
    batch_size=1,
    learning_rate=2e-4,
    max_grad_norm=1.0,

    # Data params
    sample_rate=22050,
    n_mels=128,
    n_fft=2048,
    hop_samples=300,  # Don't change this. Really.
    crop_mel_frames=24,
    unconditional = False,
    # Model params
    noise_schedule=np.linspace(1e-6, 0.01, 1000).tolist(),
)
