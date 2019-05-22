
# coding: utf-8

# coding: utf-8

# In[47]:


from __future__ import print_function, division
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import torch.utils.data as data
import matplotlib.pyplot as plt
import os.path
import sys
from torch.utils.data import Dataset
import random
from torch.autograd import Variable

f=open('tiny-imagenet-200/val/val_annotations.txt',"r")
lines=f.readlines()
result=[]
val_label =[]
val_label_dict ={}
for x in lines:
    words =x.split()
    class_label = words[1]
    image_label =words[0]
    val_label_dict[image_label]=class_label
    val_label.append(class_label)
f.close()
val_label_dict
#this is the code for triplet sampling

def has_file_allowed_extension(filename, extensions):
    
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.JPEG', '.png', '.ppm', '.bmp', '.pgm', '.tif']
class ImagenetFolder(data.Dataset):
    def __init__(self, root, loader=pil_loader, extensions=IMG_EXTENSIONS,val_label =val_label, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        samples = make_dataset(root, class_to_idx, extensions)
        #print(type(samples))
        self.root = root
        self.val_label =val_label
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    def __getitem__(self, index):
        if self.root == "tiny-imagenet-200/train":
        #print(self.samples[index])
            path1, target1 = self.samples[index]
        #print(path1)
            query_image = self.loader(path1)
        #print(query_image)
            x = self.samples.pop(index)
            path_list = path1.split(os.sep)
            path_list=path_list[:-1]
            folder_path = '/'.join(path_list)
            same_folder =[]
            for s in self.samples:
                address=s[0]
                if folder_path in address:
                    same_folder.append(address)
                else:
                    negative_image_path,negative_image_target  = random.choice(self.samples)
            positive_image =random.choice(same_folder)
#         print(positive_image)
            positive_image =self.loader(positive_image)
        #print(negative_image_path)
            negative_image = self.loader(negative_image_path)
        #print(negative_image)
            self.samples.append(x)
            if self.transform is not None:
                query_image = self.transform(query_image)
                positive_image = self.transform(positive_image)    
                negative_image = self.transform(negative_image)
            
            return query_image, positive_image, negative_image
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            path_list = path.split(os.sep)
            image_name=path_list[-1]
            #print(path_list)
            target =val_label[index]
            if self.transform is not None:
                sample = self.transform(sample)
            for key, value in val_label_dict.items():
                if image_name ==key:
                    label = value
                    print(type(label))
#             if self.target_transform is not None:
#                 label = self.target_transform(label)
#                 print(type(label))
            return sample, label
    def __len__(self):
        return len(self.samples)

def _find_classes(dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    
class ImageFolder(ImagenetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=pil_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS, val_label =val_label,
                                          transform=transform, 
                                         target_transform=target_transform)
        self.imgs = self.samples

        
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
         ])
train = ImagenetFolder(root='tiny-imagenet-200/train',transform = transform)
val = ImagenetFolder(root='tiny-imagenet-200/val',transform = transform)
train_loader =torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
val_loader =torch.utils.data.DataLoader(val, batch_size=1, shuffle=True, num_workers=4)

resnet_ft = models.resnet101(pretrained=True)
for param in resnet_ft.parameters():
      param.requires_grad = False

num_ftrs = resnet_ft.fc.in_features
resnet_ft.fc = nn.Linear(num_ftrs, 4096)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet_ft = resnet_ft.to(device)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet_ft.parameters()),
                   lr=.0001)
num_epochs =10
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (query_image,positive_image,negative_image) in enumerate(train_loader):
        data_a, data_p, data_n = Variable(query_image), Variable(positive_image), Variable(negative_image)       
        optimizer.zero_grad()
        out_a, out_p, out_n = resnet_ft(data_a), resnet_ft(data_p), resnet_ft(data_n)
        loss = triplet_loss(out_a, out_p, out_n)
        #print(loss)
        loss.backward()
        optimizer.step()        
        running_loss += loss.item()
        if batch_idx % 2 == 1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0
            
print('Finished Training')

correct = 0
total = 0


#with torch.no_grad():

resnet_ft.eval()
test_acc = 0.0
for i, batch in enumerate(val_loader):
    print(batch)
    images, labels =batch
    #print(labels)
    images = Variable(images)
    labels = variable(labels)
        # Predict classes using images from the test set
    outputs = resnet_ft(images)
    _, prediction = torch.max(outputs.data, 1)
    test_acc += torch.sum(prediction == labels)

    # Compute the average acc and loss over all 10000 test images
test_acc = test_acc / 10000

print(test_acc)


 # Save the Model
torch.save(resnet_ft.state_dict(), 'model_resnet.pkl')

