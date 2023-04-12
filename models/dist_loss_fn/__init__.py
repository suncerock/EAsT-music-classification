from .label_distillation import LabelDistillationLoss
from .feature_space_reg import FeatureSpaceRegularizationLoss
from .combine_dist import CombineDistillationLoss

ALL_LOSSES = dict(
    LabelDistillationLoss=LabelDistillationLoss,
    FeatureSpaceRegularizationLoss=FeatureSpaceRegularizationLoss,
    CombineDistillationLoss=CombineDistillationLoss
)

def get_dist_loss_fn(cfg: dict):
    return ALL_LOSSES[cfg["name"]](**cfg["args"])