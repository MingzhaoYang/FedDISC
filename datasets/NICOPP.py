from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from tqdm import tqdm
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

imgsize = 224

nicopp_path = "/home/share/NICOpp/NICO_DG/autumn"
f = os.listdir(nicopp_path)
for i in range(len(f)):
    f[i] = f[i].lower()
nicopp_class_prompts = sorted(f)  
        
def read_nicopp_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "NICO_DG_official", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            b = line.split('/')
            b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
            data_path =f"{'/'.join(b)}"
            # data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            #label = int(label)
            label = nicopp_class_prompts.index(b[-2])
            data_paths.append(data_path)
            data_labels.append(label)
    return np.array(data_paths), np.array(data_labels)

class Nicopp(Dataset):
    def __init__(self, data_paths, data_labels, transforms):
        super(Nicopp, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index] 
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data_paths)



def get_nicopp_dataset(transform,divide):
    dataset_path = '/home/share/NICOpp'
    train_data_paths, train_data_labels = read_nicopp_data(dataset_path, divide, split="train")
    test_data_paths, test_data_labels = read_nicopp_data(dataset_path, divide, split="test")

    train_dataset = Nicopp(train_data_paths, train_data_labels, transform)
    test_dataset = Nicopp(test_data_paths, test_data_labels, transform)
    
    return train_dataset, test_dataset



def read_nicou_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    class_style = {}
    for i in os.listdir('/home/share/NICOpp/txtlist/NICO_unique_official'):
        if '.DS_Store' in i: continue
        c,s = i.split('_')[0],i.split('_')[1]
        if c in class_style.keys():# and i.split('_')[2]=='test.txt':
            if s not in class_style[c]:
                class_style[c].append(s)
        else:
            class_style[c] = [s,]
    for cla in class_style.keys():
        class_style[cla] = sorted(class_style[cla])
    class_style = sorted(class_style.items(), key=lambda x: x)
    files = []
    for cla in class_style:
        c,s = cla[0],cla[1][domain_name]
        file = '/home/share/NICOpp/txtlist/NICO_unique_official/'+'_'.join([c,s])+f'_{split}.txt'
        files.append(file)

    # split_file = path.join(dataset_path, "txtlist/NICO_unique_official", "{}_{}.txt".format(domain_name, split))
    for split_file in files:
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if '.DS_Store' in line: continue
                line = line.strip()
                b = line.split('/')
                nicopp_class_prompts.index(b[-3])
                b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
                data_path =f"{'/'.join(b[4:])}"
                # data_path, label = line.split(' ')
                data_path = path.join('/home/share/NICOpp', data_path)
                #label = int(label)
                label = nicopp_class_prompts.index(b[-3])
                data_paths.append(data_path)
                data_labels.append(label)
    return np.array(data_paths), np.array(data_labels)


def get_nicou_dataset(transform,divide):
    dataset_path = '/home/share/NICOpp'
    
    train_data_paths, train_data_labels = read_nicou_data(dataset_path, divide, split="train")
    test_data_paths, test_data_labels = read_nicou_data(dataset_path, divide, split="test")
    
    train_dataset = Nicopp(train_data_paths, train_data_labels, transform)
    test_dataset = Nicopp(test_data_paths, test_data_labels, transform)
    
    return train_dataset, test_dataset
