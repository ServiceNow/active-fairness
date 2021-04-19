import argparse
import itertools
import os
import random
from copy import deepcopy

import numpy as np
import sklearn.metrics as skm
import torch
import torch.backends
from baal.active import get_heuristic
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from active_fairness import utils
from active_fairness.balanced_heuristic import BalancedHeuristic
from active_fairness.grad import GradCrit, GRADModel, GRADWrapper
from active_fairness.metrics import FairnessMetric
from active_fairness.utils import get_datasets

pjoin = os.path.join


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("attribute")
    parser.add_argument("target_key")
    parser.add_argument("--random", default=1337)
    parser.add_argument("--oracle", action='store_true')
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=500, type=int)
    parser.add_argument("--query_size", default=100, type=int)
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--heuristic", default="random", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--lambda", default=0, type=float)
    parser.add_argument('--learning_epoch', default=10, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    hyperparams = vars(args)

    seed = hyperparams['random']
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    attribute = hyperparams['attribute']
    active_set, val_set, test_set = get_datasets(hyperparams['dataset'],
                                                 hyperparams['initial_pool'],
                                                 attribute,
                                                 hyperparams['target_key'])
    num_classes = len(test_set._all_target)
    num_group = len(test_set._all_attribute)

    heuristic = get_heuristic(hyperparams['heuristic'])
    criterion = GradCrit(CrossEntropyLoss(), lmd=hyperparams['lambda'], attribute=attribute)
    model = utils.vgg16(pretrained=True, num_classes=num_classes)
    model = GRADModel(model, num_group)

    # change dropout layer to MCD
    model = patch_module(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if use_cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9,
                          weight_decay=hyperparams['weight_decay'])

    model = GRADWrapper(model, criterion)
    model.add_metric('fair_recall',
                     lambda: FairnessMetric(skm.recall_score, 'recall', average='micro',
                                            attribute=attribute))
    model.add_metric('fair_accuracy',
                     lambda: FairnessMetric(skm.accuracy_score, 'accuracy',
                                            attribute=hyperparams['attribute']))
    model.add_metric('fair_precision',
                     lambda: FairnessMetric(skm.precision_score, 'precision', average='micro',
                                            attribute=attribute))
    model.add_metric('fair_f1',
                     lambda: FairnessMetric(skm.f1_score, 'f1', average='micro',
                                            attribute=attribute))

    # save imagenet weights
    init_weights = deepcopy(model.state_dict())

    # for prediction we use a smaller batchsize
    # since it is slower
    if hyperparams['oracle']:
        loop_oracle = BalancedHeuristic(active_set, heuristic, hyperparams['attribute'])
    else:
        loop_oracle = heuristic
    active_loop = ActiveLearningLoop(active_set,
                                     model.predict_on_dataset_generator,
                                     loop_oracle,
                                     hyperparams['query_size'],
                                     batch_size=6,
                                     iterations=hyperparams['iterations'],
                                     use_cuda=use_cuda,
                                     workers=0)
    learning_epoch = hyperparams['learning_epoch']
    for epoch in tqdm(itertools.count(start=0), desc="Active loop"):
        if len(active_set) > 20000:
            break
        criterion.train()
        model.load_state_dict(init_weights)
        model.train_on_dataset(active_set, optimizer, hyperparams["batch_size"],
                               learning_epoch, use_cuda, workers=0)

        # Validation!
        criterion.eval()
        model.test_on_dataset(test_set, batch_size=6, use_cuda=use_cuda, workers=0,
                              average_predictions=hyperparams['iterations'])
        fair_logs = {}
        for met in ['fair_recall', 'fair_accuracy', 'fair_precision', 'fair_f1']:
            fair_test = model.metrics[f'test_{met}'].value
            fair_test = {'test_' + k: v for k, v in fair_test.items()}
            fair_train = model.metrics[f'train_{met}'].value
            fair_train = {'train_' + k: v for k, v in fair_train.items()}
            fair_logs.update(fair_test)
            fair_logs.update(fair_train)
        metrics = model.metrics

        should_continue = active_loop.step()
        if not should_continue:
            break

        # Send logs
        train_loss = metrics['train_loss'].value
        val_loss = metrics['test_loss'].value

        logs = {
            "test_loss": val_loss,
            "train_loss": train_loss,
            "epoch": epoch,
            "labeled_data": active_set.labelled,
            "next_training_size": len(active_set)
        }
        logs.update(fair_logs)
        print(logs)


if __name__ == "__main__":
    main()
