import numpy as np
from torch.utils.data import Dataset


class SynbolDataset(Dataset):
    def __init__(self, x, y, metadata, target_key, attribute, transform, encode_groups=False):
        self.x, self.transform = x, transform
        self.target_key = target_key
        self.attribute = attribute
        self._all_target = sorted(np.unique([d[self.target_key] for d in metadata]).tolist())
        self.encode_groups = encode_groups
        self._all_attribute = sorted(np.unique([d[self.attribute] for d in metadata]).tolist())

        self.metadata = metadata

    def get_attr(self, item, attr, index=None):
        out = self.metadata[item][attr]
        if index:
            out = index.index(out)
        return out

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        target = self.get_attr(item, self.target_key, self._all_target)
        attr = self.get_attr(item, self.attribute,
                             index=self._all_attribute if self.encode_groups else None)
        return self.transform(self.x[item]), {'target': target, self.attribute: attr}
