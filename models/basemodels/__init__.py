import torch

from .pretrained_feature_classifier import PretrainedFeatureClassifier
from .cp_resnet import CPResNet
from .mobile_fcn import MobileFCN

ALL_MODELS = dict(
    PretrainedFeatureClassifier=PretrainedFeatureClassifier,
    CPResNet=CPResNet,
    MobileFCN=MobileFCN
)
PRED_FROM_DICT = dict(
    PretrainedFeatureClassifier="feature",
    CPResNet="x",
    MobileFCN="x"
)

def get_base_model_and_pred_from(cfg: dict):
    name = cfg.get("name")
    ckpt = cfg.get("ckpt", None)
    args = cfg.get("args", {})

    model = ALL_MODELS[name](**args)
    pred_from = PRED_FROM_DICT[name]
    if ckpt is not None:
        print("Loading pretrained model from {}!".format(ckpt))
        pretrained_state_dict = torch.load(ckpt)["state_dict"]
        state_dict = dict()
        for layer_name in pretrained_state_dict:
            if not layer_name.startswith('student.'):
                continue
            new_layer_name = layer_name.replace('student.', '')
            state_dict[new_layer_name] = pretrained_state_dict[layer_name]
        model.load_state_dict(state_dict)

    return model, pred_from
