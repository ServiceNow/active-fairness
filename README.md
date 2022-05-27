# Can Active Learning Preemptively Mitigate Fairness Issues?

<p align="center">
<img src="https://user-images.githubusercontent.com/8976546/170715692-10661c82-5283-4cff-bf88-e5b58a255820.jpg" width="75%"/>
</p>
Code for the paper "Can Active Learning Preemptively Mitigate Fairness Issues?" presented at RAI 2021.

Arxiv link: https://arxiv.org/pdf/2104.06879.pdf

This repo aims at helping researchers reproduce our paper. We welcome questions and suggestions, please submit an issue! 
If you are working in active learning, our library [BaaL](https://github.com/ElementAI/baal/) might help you!


### Glossary

* query_size : Number of data labelled per AL step
* AL Step: process of training, selecting samples, and labelling.
* learning_epoch: Number of epoch to train the model before uncertainty estimation.


# Datasets

We experimented with many datasets and we have one format.

```python
# For synbols datasets
with open(dataset_path, 'rb') as f:
    train_ds = pickle.load(f)
    test_ds = pickle.load(f)
    val_ds = pickle.load(f)


x,y = train_ds
print(x[0])
# A numpy array with shape [64,64,3]

print(y[0])
# A dictionnary with all the attributes and keys.
"""
{'char':'a',
 'color': 'r',
 ...
}
"""
```

# Models

We use a VGG-16 and we do AL with MC-Dropout.

```python
from active_fairness import utils
from baal.bayesian.dropout import patch_module

hyperparams = {} # Dictionnary for the hparams

model = utils.vgg16(pretrained=hyperparams['pretrained'],
                          num_classes=hyperparams['n_cls'])

# change dropout layer to be able to use MC-Dropout
model = patch_module(model)
```

# How to launch

Here is an example to run our experiment on a particular dataset.

To generate the datasets, please look in `misc/biased_dataset.ipynb`.

`poetry run python experiments/grad_experiment.py datasets/minority_dataset_50000.pkl color char`
