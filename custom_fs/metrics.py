import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.functional import auc
import torch.nn.functional as F


# LOSS FUNCTIONS


class MaskedBCE(torch.nn.BCEWithLogitsLoss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MaskedBCE, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=self.reduction)


# METRICS


class DeltaAUPRC(BinaryPrecisionRecallCurve):

    def __init__(self):
        super(DeltaAUPRC, self).__init__()

    def compute(self, pos_label_ratio):
        prc = super().compute()
        auprc = auc(prc[1], prc[0])

        return auprc - pos_label_ratio
