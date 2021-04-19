import pickle
from functools import partial

import numpy as np
import torch
from baal.utils.metrics import Metrics
from fairlearn import metrics as flm


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array(list(x))


class FairnessMetric(Metrics):
    def __init__(self, metric, name, attribute, **kwargs):
        """
        Wrapper arround fairlearn metrics for a per-class, per-group.
        Args:
            metric (callable): metric funtion with signature  (target, pred)
            name (str): Name of the metric
            attribute (str): attribute to get from the target
            **kwargs: kwargs for the metrics. (`average` for example)
        """
        self.name = name
        self.metric = metric
        self.attribute = attribute
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def reset(self):
        self._ypred = []
        self._ypred_probs = []
        self._ytrue = []
        self._sensitive_attr = []

    def update(self, output=None, target=None):
        if isinstance(output, (tuple, list)):
            # If the model predicts multiple things, the cls is the first.
            output = output[0]
        clss, clrs = target['target'], target[self.attribute]
        self._ypred.append(np.argmax(output.detach().cpu().numpy(), 1))
        self._ypred_probs.append(output.detach().cpu().numpy())
        self._ytrue.append(clss.detach().cpu().numpy())
        self._sensitive_attr.append(to_numpy(clrs))

    def save(self, fp):
        dat = (self._ypred, self._ytrue, self._sensitive_attr)
        pickle.dump(dat, open(fp, 'wb'))

    @classmethod
    def compute_metrics(cls, name, metric, y_true, y_pred, sensitive, **kwargs):
        mets = dict()
        group_metrics = flm.MetricFrame(partial(metric, **kwargs),
                                        y_true, y_pred,
                                        sensitive_features=sensitive,
                                        )

        groups = {str(k): v for k, v in group_metrics.by_group.items()}
        print(f"Overall {name} = ", group_metrics.overall)
        print(f"{name} by groups = ", list(groups.items()))
        print(f"{name} ratio = ", group_metrics.ratio('between_groups'))
        print(f"{name} diff = ", group_metrics.difference('between_groups'))
        mets[f"{name}"] = group_metrics.overall
        mets[f"ratio_{name}"] = group_metrics.ratio('between_groups')
        mets[f"diff_{name}"] = group_metrics.difference('between_groups')

        mets[f'{name}_equalized_odds'] = flm.equalized_odds_difference(y_true, y_pred,
                                                                       sensitive_features=sensitive)
        dp_key = f'{name}_demographic_parity'
        mets[dp_key] = flm.demographic_parity_difference(y_true, y_pred,
                                                         sensitive_features=sensitive)
        for g, v in list(groups.items()):
            mets[f"{name}_{g}"] = v
        return mets

    @property
    def value(self):
        sensitive, y_pred, _, y_true = self.aggregate_data()
        mets = self.compute_metrics(self.name, self.metric, y_true, y_pred, sensitive,
                                    **self.kwargs)

        # Count
        for ch in np.unique(y_true):
            mask = y_true == ch
            colors = np.array(sensitive)[mask].copy()
            counts = np.unique(colors, return_counts=True)
            for k, c in zip(*counts):
                mets[f"count_{ch}_{k}"] = int(c)
        return mets

    def aggregate_data(self):
        y_true = np.concatenate(self._ytrue, 0).reshape([-1])
        y_pred = np.concatenate(self._ypred, 0).reshape([-1])
        y_pred_probs = np.concatenate(self._ypred_probs, 0).reshape([y_pred.shape[0], -1])
        sensitive = np.concatenate(self._sensitive_attr, 0).reshape([-1])
        return sensitive, y_pred, y_pred_probs, y_true
