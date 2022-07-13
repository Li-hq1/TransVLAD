import os
from PIL import Image
import torch
import pickle
from torchvision import transforms
from datasets.masking_generator import RandomMaskingGenerator

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.data import create_transform
import numpy as np

# -----------------------dataset------------------------ #
ROOT_DIR = "data/datasets"


class MiniImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, split, transform):
        if split == "train":
            split_tag = "train_phase_train"
        elif split == "val":
            split_tag = "train_phase_val"
        elif split == "test":
            split_tag = "train_phase_test"
        elif split == "meta_val":
            split_tag = "val"
        elif split == "meta_test":
            split_tag = "test"
        else:
            assert False, "Dataset: 'split' name is wrong"
        split_file = "miniImageNet_category_split_{}.pickle".format(split_tag)
        with open(os.path.join(root_path, split_file), "rb") as f:
            pack = pickle.load(f, encoding="latin1")

        data = pack["data"]
        label = pack["labels"]
        data = [Image.fromarray(x) for x in data]
        min_label = min(label)
        label = [x - min_label for x in label]
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class Cifar100Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, name, split, transform):
        if split == "train":
            split_tag = "train"
        elif split == "meta_val":
            split_tag = "val"
        elif split == "meta_test":
            split_tag = "test"
        else:
            assert False, "Dataset: 'split' name is wrong"
        assert name == "CIFAR_FS" or name == "FC100"
        split_file = name + "_{}.pickle".format(split_tag)
        with open(os.path.join(root_path, split_file), "rb") as f:
            pack = pickle.load(f, encoding="latin1")

        data = pack["data"]
        labels = pack["labels"]

        cur_class = 0
        label2label = {}
        for idx, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])
        data = [Image.fromarray(x) for x in data]

        self.data = data
        self.label = new_labels

        self.n_classes = len(set(self.label))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class CubDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, name, split, transform):
        if split == "train":
            split_tag = "train"
        elif split == "meta_val":
            split_tag = "val"
        elif split == "meta_test":
            split_tag = "test"
        else:
            assert False, "Dataset: 'split' name is wrong"
        assert name == "CUB"

        IMAGE_PATH = root_path
        SPLIT_PATH = os.path.join(root_path, "split/")
        txt_path = os.path.join(SPLIT_PATH, split_tag + ".csv")

        lines = [x.strip() for x in open(txt_path, "r").readlines()][1:]

        if split_tag == "train":
            lines.pop(5864)  # this image file is broken

        data = []
        labels = []
        lb = -1

        self.wnids = []

        for l in lines:
            context = l.split(",")
            name = context[0]
            wnid = context[1]
            path = os.path.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            labels.append(lb)

        cur_class = 0
        label2label = {}
        for idx, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])

        data = [Image.open(path).convert("RGB") for path in data]

        self.data = data
        self.label = new_labels

        self.n_classes = len(set(self.label))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class TieredDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, name, split, transform):
        if split == "train":
            split_tag = "train"
        elif split == "meta_val":
            split_tag = "val"
        elif split == "meta_test":
            split_tag = "test"
        else:
            assert False, "Dataset: 'split' name is wrong"
        assert name == "tiered"

        THE_PATH = os.path.join(root_path, split_tag)

        data = []
        labels = []

        folders = [
            os.path.join(THE_PATH, label)
            for label in os.listdir(THE_PATH)
            if os.path.isdir(os.path.join(THE_PATH, label))
        ]
        folders.sort()

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            this_folder_images.sort()
            for image_path in this_folder_images:
                data.append(os.path.join(this_folder, image_path))
                labels.append(idx)

        cur_class = 0
        label2label = {}
        for idx, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])

        self.data = data
        self.label = new_labels

        self.n_classes = len(set(self.label))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(Image.open(self.data[i]).convert("RGB")), self.label[i]


# sampler for few-shot learning
class CategoriesSampler:
    def __init__(self, labels, n_iteration, n_class, n_sample, n_episode=1):
        self.n_iteration = n_iteration
        self.n_class = n_class
        self.n_sample = n_sample
        self.n_episode = n_episode

        labels = np.array(labels)
        self.catlocs = []
        for c in range(max(labels) + 1):
            self.catlocs.append(np.argwhere(labels == c).reshape(-1))

    def __len__(self):
        return self.n_iteration

    def __iter__(self):
        for i_batch in range(self.n_iteration):
            batch = []
            for i_ep in range(self.n_episode):
                episode = []
                classes = np.random.choice(
                    len(self.catlocs), self.n_class, replace=False
                )
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_sample, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)  # n_class * n_sample
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_class * n_sample
            yield batch.view(-1)


# -------------------------pretrain------------------------------ #
class DataAugmentationForMAE(object):
    """
    return transformed image and masking position for image patch
    """

    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = (
            IMAGENET_INCEPTION_MEAN
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_MEAN
        )
        std = (
            IMAGENET_INCEPTION_STD
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_STD
        )

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator
        )
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    root_dir = ROOT_DIR
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    dataset_name = args.dataset_name
    if dataset_name == "mini":
        root_path = os.path.join(root_dir, "mini-imagenet")
        return MiniImagenetDataset(root_path, split="train", transform=transform)
    elif dataset_name == "CUB":
        root_path = os.path.join(root_dir, "cub")
        name = "CUB"
        return CubDataset(root_path, name, split="train", transform=transform)
    elif dataset_name == "tiered":
        root_path = os.path.join(root_dir, "tiered_imagenet")
        name = "tiered"
        return TieredDataset(root_path, name, split="train", transform=transform)
    elif dataset_name == "CIFAR_FS":
        root_path = os.path.join(root_dir, "CIFAR_FS")
        name = "CIFAR_FS"
        return Cifar100Dataset(root_path, name, split="train", transform=transform)
    elif dataset_name == "FC100":
        root_path = os.path.join(root_dir, "FC100")
        name = "FC100"
        return Cifar100Dataset(root_path, name, split="train", transform=transform)
    else:
        assert False


# -----------------------finetune------------------------ #
def build_dataset(is_train, dataset_name, split, args):
    root_dir = ROOT_DIR
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")
    if dataset_name == "mini":
        root_path = os.path.join(root_dir, "mini-imagenet")
        dataset = MiniImagenetDataset(root_path, split, transform=transform)
    elif dataset_name == "CUB":
        root_path = os.path.join(root_dir, "cub")
        name = "CUB"
        dataset = CubDataset(root_path, name, split, transform=transform)
    elif dataset_name == "tiered":
        root_path = os.path.join(root_dir, "tiered_imagenet")
        name = "tiered"
        dataset = TieredDataset(root_path, name, split, transform=transform)
    elif dataset_name == "CIFAR_FS":
        root_path = os.path.join(root_dir, "CIFAR_FS")
        name = "CIFAR_FS"
        dataset = Cifar100Dataset(root_path, name, split, transform=transform)
    elif dataset_name == "FC100":
        root_path = os.path.join(root_dir, "FC100")
        name = "FC100"
        dataset = Cifar100Dataset(root_path, name, split, transform=transform)
    else:
        assert False
    nb_classes = dataset.n_classes
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = (
        IMAGENET_INCEPTION_MEAN
        if not imagenet_default_mean_and_std
        else IMAGENET_DEFAULT_MEAN
    )
    std = (
        IMAGENET_INCEPTION_STD
        if not imagenet_default_mean_and_std
        else IMAGENET_DEFAULT_STD
    )

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(
                size, interpolation=transforms.functional.InterpolationMode.BICUBIC
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
