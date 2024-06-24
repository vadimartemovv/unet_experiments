import torch
import torchmetrics
from torchmetrics.classification import MulticlassJaccardIndex

class MeanIoU(torchmetrics.Metric):
    def __init__(self, num_classes, ignore_index=-1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.jaccard = MulticlassJaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index)
        self.add_state("total_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Create a mask to ignore the specified index
        preds = torch.argmax(preds, axis=1)
        mask = target != self.ignore_index
        masked_preds = preds[mask]
        masked_target = target[mask]

        # Check if there's any valid data left after masking
        if masked_target.numel() > 0:
            iou = self.jaccard(masked_preds, masked_target)
            self.total_iou += iou
            self.num_batches += 1

    def compute(self):
        MIOU =  self.total_iou / self.num_batches
        return MIOU.item()
    
def accuracy_ignore_negative_one(preds, targets):
    """
    Compute the accuracy, ignoring target indices with a value of -1.

    Args:
        preds (torch.Tensor): Predictions from the model, shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels, shape (batch_size).

    Returns:
        float: Accuracy score.
    """
    # Ensure the predictions are in the form of class indices
    _, predicted_labels = torch.max(preds, dim=1)
    
    # Filter out indices where the target is -1
    valid_indices = targets != -1
    valid_targets = targets * valid_indices
    valid_predictions = predicted_labels * valid_indices

    # Compute the accuracy
    correct = valid_predictions.eq(valid_targets).sum().item()
    total = valid_targets.size()[0]*valid_targets.size()[1]*valid_targets.size()[2]
    # total = valid_targets.size().sum()
    #print(correct, total)
    
    if total == 0:
        return 0.0  # If no valid targets, return 0.0 accuracy

    accuracy = correct / total
    return accuracy