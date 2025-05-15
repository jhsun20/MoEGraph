import torch
from graphmetro.datasets.good import *  # Assumes good dataset classes are here
from graphmetro.config import cfg       # Config with shift_type and seed

def load_good_dataset(name, root="/path/to/data"):
    assert name.startswith("GOOD"), "Only GOOD datasets are supported"
    class_name = eval(f'GOOD{name[4:]}')  # e.g., GOODCora, GOODArxiv
    datasets, meta_info = class_name.load(
        dataset_root=root,
        shift=cfg.dataset.shift_type
    )

    if meta_info.model_level == 'node':
        for split in ['train', 'val', 'test']:
            ds = datasets
            ds.data.y = ds.data.y.view(-1).long()
            ds.n_classes = len(torch.unique(ds.data.y))
        return {'train': ds, 'val': ds, 'test': ds}
    else:
        for key, ds in datasets.items():
            if key in ['task', 'metric']:
                continue
            ds.data.y = ds.data.y.view(-1).long()
            if datasets['task'] == 'Binary classification':
                ds.n_classes = ds.data.y.size(-1)
            elif datasets['task'] == 'Regression':
                ds.n_classes = 1
            else:
                ds.n_classes = len(torch.unique(ds.data.y))
        return datasets
