import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight  = torch.tensor([1, 3061 / 669, 3061 / 105, 3061 / 2868, 3061 / 219, 3061 / 61, 3061 / 653, 3061 / 230]).to(device)

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        #if self.weight is not None:
        #    self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)

class FocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        self.alpha = Variable(torch.tensor([1, 3061 / 669, 3061 / 105, 3061 / 2868, 3061 / 219, 3061 / 61, 3061 / 653, 3061 / 230])).to(device)
        #self.alpha = Variable(torch.tensor([7866 / (8 * 3061), 7866 / (8 * 669), 7866 / (8 * 105), 7866 / (8 * 2868), 7866 / (8 * 219), 7866 / (8 * 61) , 7866 /(8 * 653) , 7866 / (8 * 230)]))

        self.gamma = gamma  # Index
        self.class_num = class_num  # Number of categories
        self.size_average = size_average  # Does the returned loss need to mean?

    def forward(self, inputs, targets):
        N = inputs.size(0)  # inputs is the output of the top layer of the network
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)  # Seek p_t first

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # Get the one_hot encoding of label

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        # y*p_t If * is not used here, you can also use gather to extract the probability of the correct category.
        # The reason why sum can be used is because class_mask has cleared the probability of prediction error to zero.
        probs = (P * class_mask).sum(1).view(-1, 1)
        # y*log(p_t)
        log_p = probs.log()
        # -a * (1-p_t)^2 * log(p_t)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class HS_LR(nn.Module):
    """Implementation of Focus rectified logistic regression loss with Hard Selection(HS-LR)
        Args:
        alpha: cost-sensitive hyperparameter in Eq. (5);
        topratio: m % in Eq. (5) for hard selection of most confusing negative classes.
    """
    def __init__(self, num_classes, alpha, topratio, use_gpu=True):
        super(HS_LR, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
        self.topratio = topratio

    def forward(self, inputs, targets):
        """
        Args:
            inputs: outputs of the last FC layer, with shape [batch_size, num_classes] (without normalisation)
            targets: ground truth labels with shape (num_classes)
        """
        eps = 1e-7
        probs = self.sigmoid(inputs)
        # Note that for the multi-label case, there is no need to convert scalar label to one-hot label.
        # Comment this line.
        targets = torch.zeros(probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)#one hot embedding

        if self.use_gpu: targets = targets.cuda()
        pred_p = torch.log(probs+eps).cuda()
        pred_n = torch.log(1.0-probs+eps).cuda()


        topk = int(self.num_classes * self.topratio)
        targets = targets.cuda()
        count_pos = targets.sum().cuda()
        hard_neg_loss = -1.0 * (1.0-targets) * pred_n
        topk_neg_loss = -1.0 * hard_neg_loss.topk(topk, dim=1)[0]#topk_neg_loss with shape batchsize*topk

        loss = (targets * pred_p).sum() / count_pos + self.alpha*(topk_neg_loss.cuda()).mean()

        return -1.0*loss



class SS_LR(nn.Module):
    """Implementation of Focus rectified logistic regression loss with Soft Selection(SS-LR)
        Args:
        alpha: cost-sensitive hyperparameter in Eq. (6);
        gamma: gamma in Eq. (6), the temperature of prediction probability in soft selection.
    """
    def __init__(self, num_classes, alpha, gamma, use_gpu=True):
        super(SS_LR, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        eps = 1e-7
        probs = self.sigmoid(inputs)
        # Note that for the multi-label case, there is no need to convert scalar label to one-hot label.
        # Comment this line.
        targets = torch.zeros(probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        count_pos = targets.sum().cuda()
        count_neg = (1.0-targets).sum().cuda()
        if self.use_gpu: targets = targets.cuda()
        pred_p = torch.log(probs+eps).cuda()
        pred_n = (torch.pow(probs, self.gamma) * torch.log(1.0-probs+eps)).cuda()

        loss = (targets * pred_p).sum() / count_pos + self.alpha*(((1.0-targets) * pred_n).sum() / count_neg)

        return - 1.0 * loss