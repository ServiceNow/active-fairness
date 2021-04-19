import torch
from baal import ModelWrapper
from baal.utils.cuda_utils import to_cuda
from pytorch_revgrad import RevGrad
from torch import nn


class GradCrit(nn.Module):
    def __init__(self, crit, lmd, attribute):
        super().__init__()
        self.crit = crit
        self.lmb = lmd
        self.attribute = attribute

    def forward(self, input, target):
        if self.training:
            cls_pred, group_pred = input
            cls_loss = self.crit(cls_pred, target['target'])
            group_loss = self.crit(group_pred, target[self.attribute])
            return cls_loss + self.lmb * group_loss
        else:
            return self.crit(input, target['target'])


class GRADWrapper(ModelWrapper):
    def predict_on_batch(self, data, iterations=1, cuda=False):
        out = super().predict_on_batch(data, iterations, cuda)
        # Return clss
        return out[0]

    def train_on_batch(self, data, target, optimizer, cuda=False,
                       regularizer=None):
        if cuda:
            data, target = to_cuda(data), to_cuda(target)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)

        if regularizer:
            regularized_loss = loss + regularizer()
            regularized_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        optimizer.step()
        self._update_metrics(output, target, loss, filter='train')
        return loss


class GRADModel(nn.Module):
    def __init__(self, model, num_groups):
        super().__init__()
        self.model = model
        self.num_groups = num_groups
        self.group_pred = nn.Sequential(
            RevGrad(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_groups)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.model.classifier(x)
        x2 = self.group_pred(x)
        return x1, x2
