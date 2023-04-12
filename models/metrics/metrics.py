import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_auroc, binary_average_precision


class MultiLabelBinaryEval(Metric):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.logits = {i: [] for i in range(self.num_classes)}
        self.target = {i: [] for i in range(self.num_classes)}

    def update(self, logits, target, mask):
        with torch.no_grad():
            for i in range(self.num_classes):
                idx = mask[:, i]

                if sum(idx) == 0:
                    continue

                self.logits[i].append(logits[idx, i])
                self.target[i].append(target[idx, i])

    def compute(self):
        with torch.no_grad():
            logits = [torch.concat(self.logits[i]) for i in range(self.num_classes)]
            targets = [torch.concat(self.target[i]) for i in range(self.num_classes)]

            f1 = [self._compute_f1(logits[i], targets[i].int()) for i in range(self.num_classes)]
            binary_f1 = [x[0] for x in f1]
            macro_f1 = [x[1] for x in f1]

            mAP = [binary_average_precision(logits[i], targets[i].int()) for i in range(self.num_classes)]
            auc_roc = [binary_auroc(logits[i], targets[i].int()) for i in range(self.num_classes)]

            return dict(
                binary_f1=sum(binary_f1) / len(binary_f1),
                macro_f1=sum(macro_f1) / len(macro_f1),
                mAP=sum(mAP) / len(mAP),
                auc_roc=sum(auc_roc) / len(auc_roc),
            )

    def reset(self) -> None:
        super().reset()
        self.logits = {i: [] for i in range(self.num_classes)}
        self.target = {i: [] for i in range(self.num_classes)}

    def _compute_f1(self, logits, target, threshold=0.4):
        # Hard coding for threshold
        # F1 only used on OpenMIC, not used on MagnaTagATune
        tp = torch.count_nonzero((logits > threshold)  & (target == 1))
        fp = torch.count_nonzero((logits > threshold)  & (target == 0))
        tn = torch.count_nonzero((logits <= threshold) & (target == 0))
        fn = torch.count_nonzero((logits <= threshold) & (target == 1))

        precision_p = tp / (tp + fp)
        recall_p = tp / (tp + fn)
        f1_p = 2 * precision_p * recall_p / (precision_p + recall_p)
        f1_p = torch.nan_to_num(f1_p, 0.)
        
        precision_n = tn / (tn + fn)
        recall_n = tn / (tn + fp)
        f1_n = 2 * precision_n * recall_n / (precision_n + recall_n)
        return f1_p, (f1_p + f1_n) / 2
