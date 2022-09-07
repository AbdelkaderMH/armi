import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics import accuracy_score, f1_score


def label_rule(mis_preds, cat_preds):
    for i in range(len(mis_preds)):
        #print(mis_preds[i], cat_preds[i])
        if mis_preds[i] == 0:
            cat_preds[i] = 0
    return cat_preds


def accuracy(preds, y):
    all_output = preds.float().cpu()
    all_label = y.float().cpu()
    _, predict = torch.max(all_output, 1)
    acc = accuracy_score(all_label.numpy(), torch.squeeze(predict).float().numpy())
    return acc

def calc_accuracy(preds,y):
    predict = torch.argmax(preds, dim=1)
    accuracy = torch.sum(predict == y.squeeze()).float().item()
    return accuracy / float(preds.size()[0])

def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds).squeeze()

    correct = (rounded_preds == y).float()
    acc = correct.sum() / y.size(0)
    return acc

def fscore_loss(y_pred, y_true, epsilon=1e-8):
    assert y_pred.ndim == 2
    assert y_true.ndim == 1
    y_true = F.one_hot(y_true, 8).to(torch.float32)
    y_pred = F.softmax(y_pred, dim=1)

    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)
    return 1 - f1.mean()


def macro_double_soft_f1(y, y_hat):
    pred = y_hat.to(torch.float).unsqueeze(1)
    y = F.one_hot(y, num_classes=8).float()
    truth = y.to(torch.float).unsqueeze(1)
    tp = pred.mul(truth).sum(0).float()
    fp = pred.mul(1 - truth).sum(0).float()
    fn = (1 - pred).mul(truth).sum(0).float()
    tn = (1 - pred).mul(1 - truth).sum(0).float()
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = cost.mean()  # average on all labels
    return macro_cost


def mcc_loss(outputs_target, temperature=2.8, class_num=8):
    train_bs = outputs_target.size(0)
    outputs_target_temp = outputs_target / temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    target_entropy_weight = Entropy(target_softmax_out_temp).detach()
    target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
    target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
    cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(target_softmax_out_temp)
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
    return mcc_loss

def EntropyLoss(input_):
    # print("input_ shape", input_.shape)
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out + 1e-5)))
    return entropy / float(input_.size(0))

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy