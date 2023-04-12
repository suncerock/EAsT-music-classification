from .waveform_featurizer import WaveformFeaturizer
from .log_mel_featurizer import LogMelFeaturizer

_featurizers_dict = dict(
    waveform_featurizer=WaveformFeaturizer,
    log_mel_featurizer=LogMelFeaturizer
)

def build_featurizer(cfg, training=True):
    featurizer_name = cfg.get("name").lower()

    if featurizer_name in _featurizers_dict:
        Featurizer = _featurizers_dict[featurizer_name]
    else:
        raise KeyError("Expect model name in {}, but got {}!".format(_featurizers_dict.keys(), featurizer_name))

    featurizer = Featurizer(training=training, **cfg.get('args'))
    return featurizer