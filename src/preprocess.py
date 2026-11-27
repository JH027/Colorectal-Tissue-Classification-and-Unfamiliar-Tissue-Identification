import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


#predetermined values that torchvision expects to use for normalization of inputs
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train):
    norm = T.Normalize(MEAN, STD)
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.ToTensor(), norm])


def get_classes(root):
    """looks inside dataset folder and returns sorted names of subfolders/tissue classes"""
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())


def list_images(root, class_name):
    """returns sorted list of .tif image paths inside one class folder"""
    return sorted((Path(root) / class_name).glob("*.tif"))


def make_splits(root, held_out, val_frac=0.2, seed=42):
    """main splitting func --> returns train, val, and lists of (path, label) and the known class list"""
    """drops held out class, gives each remaining class number 0 to 7, shuffles each class's images w/"""
    """fixed seed, 20% set for validation"""
    known = [c for c in get_classes(root) if c != held_out]
    label_of = {c: i for i, c in enumerate(known)}
    rng = random.Random(seed)
    train, val = [], []
    for c in known:
        paths = list_images(root, c)
        rng.shuffle(paths)
        n_val = int(len(paths) * val_frac)
        val += [(p, label_of[c]) for p in paths[:n_val]]
        train += [(p, label_of[c]) for p in paths[n_val:]]
    return train, val, known


def make_test_samples(test_root, known, held_out):
    """known classes get labels 0..7; the held-out class gets -1 (unknown)"""
    label_of = {c: i for i, c in enumerate(known)}
    samples = []
    for c in get_classes(test_root):
        label = label_of.get(c, -1) if c != held_out else -1
        samples += [(p, label) for p in list_images(test_root, c)]
    return samples

def make_subset(samples, n_per_class, seed=42):
    """keeps at most n_per_class random images from each label, so every class stays represented"""
    #this to help w/ navigating issue w/ large dataset --> mebbe take way too long to run :')
    rng = random.Random(seed)
    by_label = {}
    for path, label in samples:
        by_label.setdefault(label, []).append((path, label))
    subset = []
    for items in by_label.values():
        rng.shuffle(items)
        subset += items[:n_per_class]
    rng.shuffle(subset)
    return subset


class CRCDataset(Dataset):
    def __init__(self, samples, transform):
        """stores samples and the transform"""
        self.samples = samples
        self.transform = transform

    def __len__(self):
        """num of imgs"""
        return len(self.samples)

    def __getitem__(self, i):
        """opens image i, converts to RGB, applies transform, return img tensor w/ label"""
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_loader(samples, train, batch_size=64, num_workers=2):
    """wraps dataset in a DataLoader, which hands the model imgs in batches of 64"""
    #if train = true --> shuffles order every epoch
    #if false --> keeps a fixed order
    ds = CRCDataset(samples, get_transforms(train))
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers)