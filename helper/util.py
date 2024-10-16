from __future__ import print_function

import torch
import numpy as np



class Spearman(object):
    def __init__(self, n) -> None:
        self.factor =n*(n*n-1)
        self.index = torch.tensor([i for i in range(n)])
    def __call__(self,preds, teacher_preds):
        indexS = torch.argsort(torch.argsort(preds, dim=1), dim=1)
        indexT = torch.argsort(torch.argsort(teacher_preds, dim=1), dim=1)
        loss = 1 - 6*torch.sum((indexT-indexS)**2, dim=1)/self.factor
        return loss


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
    # if True:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        if opt.JPEG_enable:
            new_lr_JPEG = opt.JEPG_learning_rate * (opt.lr_decay_rate ** steps)
            optimizer.param_groups[0]['lr'] = new_lr
            optimizer.param_groups[1]['lr'] = new_lr_JPEG
            optimizer.param_groups[2]['lr'] = new_lr_JPEG
            # optimizer.param_groups[2]['lr'] = new_lr_JPEG/np.sqrt(2)
            if opt.JPEG_alpha_trainable: 
                alpha_learning_rate = opt.alpha_learning_rate * (opt.lr_decay_rate ** steps)
                optimizer.param_groups[3]['lr'] = alpha_learning_rate
                optimizer.param_groups[4]['lr'] = alpha_learning_rate
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

def adjust_learning_rate_JPEG(epoch, opt, optimizer_underlying_model, optimizer_JPEG_Layer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        if opt.JPEG_enable:
            new_lr_JPEG = opt.JEPG_learning_rate * (opt.lr_decay_rate ** steps)
            optimizer_underlying_model.param_groups[0]['lr'] = new_lr
            if not opt.ADAM_enable:
                optimizer_JPEG_Layer.param_groups[0]['lr'] = new_lr_JPEG
                optimizer_JPEG_Layer.param_groups[1]['lr'] = new_lr_JPEG
            if opt.JPEG_alpha_trainable and not opt.ADAM_enable: 
                alpha_learning_rate = opt.alpha_learning_rate * (opt.lr_decay_rate ** steps)
                optimizer_JPEG_Layer.param_groups[2]['lr'] = alpha_learning_rate
                optimizer_JPEG_Layer.param_groups[3]['lr'] = alpha_learning_rate


def adjust_learning_rate_imagenet(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
       See: https://github.com/pytorch/examples/blob/3970e068c7f18d2d54db2afee6ddd81ef3f93c24/imagenet/main.py#L404 """
    new_lr = opt.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def is_correct_prediction(output, target):
    """Binary vector with [0, 1] where correct/incorrect predictions"""
    with torch.no_grad():
        _, pred = torch.max(output, dim=1)
        v = pred.eq(target).float()
        return v


if __name__ == '__main__':

    pass