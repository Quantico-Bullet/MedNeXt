import torch
from torch import nn

from nnunet_mednext.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet_mednext.training.loss_functions.dice_loss import SoftDiceLoss

class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[[1,2,1], [2,4,2], [1,2,1]],[[2,4,2], [4,8,4], [2,4,2]], [[1,2,1], [2,4,2], [1,2,1]]]) / 8.0
        padding_out = torch.zeros((target.shape[0], target.shape[-3]+2, target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1, 1:-1] = target
        h, w, d = 3, 3, 3

        Y = torch.zeros((padding_out.shape[0], 
                         padding_out.shape[1] - h + 1, 
                         padding_out.shape[2] - w + 1, 
                         padding_out.shape[2] - d + 1
                         )).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :, :] = torch.conv3d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target[:,0]) # Target is already one-hot encoded

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes
    
class BoundaryDoU_CE_Loss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoU_CE_Loss, self).__init__()

        self.b_dou = BoundaryDoULoss(n_classes)
        self.ce = RobustCrossEntropyLoss()
    
    def forward(self, net_output, target):
        b_dou_loss = self.b_dou(net_output, target)
        ce_loss = self.ce(net_output, target[:, 0].long())

        return ce_loss + b_dou_loss
    
class BoundaryDoU_Dice_Loss(nn.Module):
    def __init__(self, n_classes, soft_dice_kwargs):
        super(BoundaryDoU_Dice_Loss, self).__init__()

        self.b_dou = BoundaryDoULoss(n_classes)
        self.dice = SoftDiceLoss(**soft_dice_kwargs)
    
    def forward(self, net_output, target):
        b_dou_loss = self.b_dou(net_output, target)
        dice_loss = self.dice(net_output, target)

        return dice_loss + b_dou_loss
    
class DoU_Dice_CE_Loss(nn.Module):
    def __init__(self, n_classes, ce_kwargs, soft_dice_kwargs, loss_weights = [0.2, 1.0, 1.0]):
        super(DoU_Dice_CE_Loss, self).__init__()

        self.loss_0 = BoundaryDoULoss(n_classes)
        self.loss_1 = SoftDiceLoss(**soft_dice_kwargs)
        self.loss_2 = RobustCrossEntropyLoss(**ce_kwargs)
        self.lw = loss_weights
    
    def forward(self, net_output, target):
        loss_0 = self.loss_0(net_output, target)
        loss_1 = self.loss_1(net_output, target)
        loss_2 = self.loss_2(net_output, target[:, 0].long())
        
        if isinstance(self.lw, int):
            return self.lw * (loss_0 + loss_1 + loss_2)

        else:
            lw_0, lw_1, lw_2 = self.lw
            return lw_0 * loss_0 + lw_1 * loss_1 + lw_2 * loss_2