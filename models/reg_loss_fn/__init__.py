from .kd import KDLoss
from .feature_space_reg import FeatureSpaceRegularizationLoss
from .combine_dist_reg import CombineDistRegLoss

ALL_LOSSES = dict(
    KDLoss=KDLoss,
    FeatureSpaceRegularizationLoss=FeatureSpaceRegularizationLoss,
    CombineDistRegLoss=CombineDistRegLoss
)

def get_reg_loss_fn(cfg: dict):
    return ALL_LOSSES[cfg["name"]](**cfg["args"])