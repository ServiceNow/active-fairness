# Can Active Learning Preemptively Mitigate Fairness Issues?

Code for the paper "Can Active Learning Preemptively Mitigate Fairness Issues?" presented at RAI 2021.
Arxiv link: https://arxiv.org/pdf/2104.06879.pdf

### Glossary

* query_size : Number of data labelled per AL step
* AL Step: process of training, selecting samples, and labelling.
* max_sample: Make the pool smaller by randomly selecting it.
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

# change dropout layer to MCD
model = patch_module(model)
```

# How to launch

Here is an example to run our experiment on a particular dataset.

To generate the datasets, please look in `misc/biased_dataset.ipynb`.

`poetry run python experiments/grad_experiment.py datasets/minority_dataset_50000.pkl color char`
