import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, n_classes=5, smoothing=1e-7) -> None:
        super(DiceLoss, self).__init__()
        self.eps = smoothing
        self.nclasses = n_classes
    
    
    def dice_loss(self, true, pred):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = pred.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(pred)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(pred, dim=1)
        true_1_hot = true_1_hot.type(pred.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)
    
    
    def dice_score(self, yt, yp):
        yp = torch.argmax(yp, dim=1).float().squeeze()
        yt = torch.argmax(yt, dim=1).float().squeeze()
        dice = 0
        for i in range(self.nclasses):
            mask_yt = (yt == i)
            mask_yp = (yp == i)
            inter = mask_yt * mask_yp
            intersection = torch.sum(torch.sum(inter, axis=1), axis=1)
            dice += (2. * intersection * self.smooth) / (torch.sum(torch.sum(mask_yt, axis=1), axis=1) + torch.sum(torch.sum(mask_yp, axis=1), axis=1) + self.smooth)
        return torch.mean(dice)
    
    
    def diceScore_multilabel(self, y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int = 5, smooth: float = 0.0001):
        y_pred = torch.argmax(y_pred, dim=1).float()
        y_pred = y_pred.unsqueeze(1)
        y_true = torch.argmax(y_true, dim=1).float().unsqueeze(1)
        
        dices = []
        for yt, yp in zip(torch.split(y_true, 1), torch.split(y_pred, 1)):
            dice = 0
            yt, yp = yt.squeeze().cpu().numpy(), yp.squeeze().cpu().numpy()
            for i in range(n_classes):
                mask_yt = (yt == i)
                mask_yp = (yp == i)
                intersection = np.sum(mask_yt * mask_yp)
                dice += (2. * intersection * smooth) / (np.sum(mask_yt) + np.sum(mask_yp) + smooth)
                # print(f"Index: {i}\n\tYT pixels: {np.sum(mask_yt)}\n\tYP pixels: {np.sum(mask_yp)}\n\tIntersect: {intersection}\n\tDice: {dice}")
            dices.append(dice/n_classes)
                
        # print(dices)
        #return np.array(dices).mean()
        return torch.Tensor(dices).mean()
        
    def forward(self, pred, target):
        #x = self.diceScore_multilabel(target, pred, self.nclasses, self.smooth)
        x = self.dice_loss(target, pred)
        #x = torch.Tensor(x)
        return x