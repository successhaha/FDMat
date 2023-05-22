from torchvision import models, utils, datasets, transforms
import os
import requests
import zipfile
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
class TinyImageNet_load(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


def TinyImageNet(data_path, downsize=True):
    if not os.path.exists(os.path.join(data_path, "tiny-imagenet-200")):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"  # 248MB
        print("Downloading Tiny-ImageNet")
        r = requests.get(url, stream=True)
        with open(os.path.join(data_path, "tiny-imagenet-200.zip"), "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print("Unziping Tiny-ImageNet")
        with zipfile.ZipFile(os.path.join(data_path, "tiny-imagenet-200.zip")) as zf:
            zf.extractall(path=data_path)

    channel = 3
    im_size = (32, 32) if downsize else (64, 64)
    num_classes = 200
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2770, 0.2691, 0.2821)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if downsize:
        transform = transforms.Compose([transforms.Resize(32), transform])

    dst_train = datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/train'), transform=transform)
    # dst_test = datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/test'), transform=transform)

    class_names = dst_train.classes

    # trainset = datasets.ImageFolder(root=os.path.join(data_dir, data, 'tiny-imagenet-200/train'), transform=transform_train)
    dst_test= TinyImageNet_load('./data/tiny-imagenet-200/', train=False, transform=transform)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
