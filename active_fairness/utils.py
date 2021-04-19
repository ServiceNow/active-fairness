import os
import pickle

from baal.active import ActiveLearningDataset
from torch.utils import model_zoo
from torchvision.models import VGG
from torchvision.models.vgg import make_layers, model_urls
from torchvision.transforms import transforms

from active_fairness.dataset import SynbolDataset

pjoin = os.path.join
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        'M',
    ],
    'E': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        512,
        'M',
    ],
}


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model_dir = '/app/model' if os.path.exists('/app/model') else None
        d = model_zoo.load_url(model_urls['vgg16'], model_dir=model_dir)
        if kwargs['num_classes'] != 1000:
            d = {k: v for k, v in d.items() if 'classifier' not in k}
        model.load_state_dict(d, strict=False)
    return model


def get_datasets(dataset, initial_pool, attribute, target_key):
    """
    Get the dataset for the experiment.
    Args:
        dataset: A path to a pickle file.
        initial_pool: Number of labels to start with
        attribute: Which attribute is the sensitive one
        target_key: Which attribute if the target

    Returns:
        ActiveSet, val_set, test set
    """
    IMG_SIZE = 64
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((IMG_SIZE, IMG_SIZE)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010))])
    is_synbol = isinstance(dataset, str) and os.path.isfile(dataset)
    assert is_synbol, "Can't find the dataset!"

    # We expect a single pickle with both dataset in it.
    with open(dataset, 'rb') as f:
        train_ds = pickle.load(f)
        test_ds = pickle.load(f)
        try:
            val_ds = pickle.load(f)
            val_set = SynbolDataset(*val_ds, target_key=target_key, attribute=attribute,
                                    transform=test_transform, encode_groups=True)
        except EOFError:
            # No val set
            val_set = None
    ds = SynbolDataset(*train_ds, target_key=target_key, attribute=attribute,
                       transform=transform, encode_groups=True)

    test_set = SynbolDataset(*test_ds, target_key=target_key, attribute=attribute,
                             transform=test_transform, encode_groups=True)

    active_set = ActiveLearningDataset(ds, pool_specifics={'transform': test_transform})
    active_set.label_randomly(initial_pool)
    return active_set, val_set, test_set
