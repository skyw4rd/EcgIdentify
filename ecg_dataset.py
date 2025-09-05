import os
import re
import random

from torch.utils import data

from collections import Counter
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image
from torchvision import transforms
import copy


# IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class EcgImage(data.Dataset):
    def __init__(self, root, transform=None, train=True, test=False, triplet_batch=(8, 4)):
        self.test = test
        self.root = root
        self.transform = transform

        self.batch_classes_num, self.classes_img_num = triplet_batch

        print('root:', self.root)
        
        classes, class_to_idx = self.find_classes(self.root)
        

        # 优化batch的选择
        self.priority_class_counters = {class_index : Counter() for class_index in sorted(class_to_idx.values())}

        self.samples_dict = self.make_dataset(self.root, class_to_idx)
        self.samples = []
        # train_samples, val_samples = self.divide_samples(self.samples_dict)
        self.set_samples()

        # if self.test:
            # self.samples = []
            # for class_idx in sorted(self.samples_dict.keys()):
                # for path in self.samples_dict[class_idx]:
                    # self.samples.append((path, class_idx))
        # elif train:
            # self.samples = train_samples
        # else:
            # self.samples = val_samples

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in self.samples]
    
    def empty_counters(self):
        for ct in self.priority_class_counters.values():
            ct.clear()
    
    def set_samples(self):
        self.samples = []
        tmp_dict = copy.deepcopy(self.samples_dict)
        while len(tmp_dict) > 0:
            choose_classes = []
            if len(tmp_dict.keys()) >= self.batch_classes_num:
                choose_classes = random.sample(tmp_dict.keys(), self.batch_classes_num)
            elif len(tmp_dict.keys()) > 0:
                choose_classes = list(tmp_dict.keys())
            else:
                break

            for class_idx in choose_classes:
                random.shuffle(tmp_dict[class_idx])
                for _ in range(self.classes_img_num):
                    if len(tmp_dict[class_idx]) > 0:
                        self.samples.append((tmp_dict[class_idx].pop(), class_idx))
                if len(tmp_dict[class_idx]) == 0:
                    del tmp_dict[class_idx]
    
    def get_triplet_batchsize(self):
        return self.triplet_batchsize
    
    def get_samples(self):
        return self.samples
    
    def get_samples_dict(self):
        return self.samples_dict
    
    def get_class_to_idx(self):
        return self.class_to_idx

    def get_nb_classes(self):
        return len(self.class_to_idx.keys())
    
    def divide_samples(self, samples_dict: Dict[int, List[str]]) -> List[Tuple[str, int]]:
        train_samples, val_samples = [], []
        for class_idx in sorted(samples_dict.keys()):
            val_num = int(0.2 * len(samples_dict[class_idx]))
            train_num = len(samples_dict[class_idx]) - val_num
            img_paths = samples_dict[class_idx]
            s = 0
            while val_num > 0:
                val_samples.append((img_paths[s], class_idx))
                s += 1
                val_num -= 1
            while train_num > 0:
                train_samples.append((img_paths[s], class_idx))
                s += 1
                train_num -= 1
        return train_samples, val_samples
        
    
    def find_classes(self, root) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {root}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def make_dataset(self, directory: str, class_to_idx: Optional[Dict[str, int]] = None) -> Dict[int, List[str]]:

        instances = {class_index : [] for class_index in sorted(class_to_idx.values())}
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    instances[class_index].append(path)
        return instances
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
            # target = self.target_transform(target)
        return sample, target
    
    def __len__(self) -> int:
        return len(self.samples)

def build_dataset(args):
    batch_classes_num, batch_img_num = args.batch_classes_num, args.batch_size // args.batch_classes_num
    data_transform = transforms.Compose([
        transforms.Resize([args.input_size, args.input_size]),
        transforms.ToTensor(),
    ])

    dataset = EcgImage(
        root=args.data_path + 'train',
        transform=data_transform,
        triplet_batch=(batch_classes_num, batch_img_num))
    
    return dataset, dataset.get_nb_classes() 
    

# def select_triplet(directory: str):

# class Triplet(data.Dataset):
    # def __init__(self, root, num_cls):