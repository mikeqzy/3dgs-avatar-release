from .zjumocap import ZJUMoCapDataset
from .people_snapshot import PeopleSnapshotDataset

def load_dataset(cfg, split='train'):
    dataset_dict = {
        'zjumocap': ZJUMoCapDataset,
        'people_snapshot': PeopleSnapshotDataset,
    }
    return dataset_dict[cfg.name](cfg, split)
