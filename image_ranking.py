
# coding: utf-8

import os
import sys
import time
import random
import zipfile

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
# from IPython.display import Image, display, HTML
#
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread


# mkdir data

# Extracting data from zipfile
# PATH=os.path.join(os.getcwd(),'tiny-imagenet-200.zip')
# DIST=os.path.join(os.getcwd(),'data')

# with zipfile.ZipFile(PATH) as zf:
#     zf.extractall(DIST)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


train_transform = transforms.Compose([transforms.Resize(size=(224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize])


test_transform = transforms.Compose([transforms.Resize(size=(224,224)),
                                     transforms.ToTensor(),
                                     normalize])

# Hyper parameters
num_epochs = 10
num_classes = 200
batch_size = 16
learning_rate = 0.001

def classes_to_id(folder):
    folder = os.path.join(folder, 'train')
    classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    classes.sort()
    class_dict = {classes[i]: i for i in range(len(classes))}
    return classes, class_dict

def load_images(folder, class_dict):
    folder = os.path.join(folder, 'train')
    image_label = []
    folder = os.path.expanduser(folder)
    for target in sorted(os.listdir(folder)):
        d = os.path.join(folder, target, 'images')
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):

                path = os.path.join(root, fname)
                item = (path, class_dict[target])
                image_label.append(item)
#     print(image_label[0])
    return image_label


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class TinyImageNetData(Dataset):
    def __init__(self, root, loader=pil_loader, transform=None, train=True):

        classes, class_dict = classes_to_id(root)

        self.root = root
        self.loader = loader
        self.transform = transform
        self.classes = classes
        # Classes to folder labels mapping
        self.class_to_idx = class_dict
        self.train = train

        if self.train:
            self.samples = load_images(root, class_dict)

            self.n = len(self.samples)

        else:
            test_path = os.path.join(self.root, 'val')
            self.samples = []
            with open(os.path.join(test_path, "val_annotations.txt")) as f:
                for line in f:
                    info = line.split("\t")
                    image_path = os.path.join(test_path, "images", info[0])
                    self.samples.append((image_path, self.class_to_idx[info[1]]))

    def __getitem__(self, index):

        if self.train:

            # Choose a random sample
            path, target = self.samples[index]
            positive = random.randint(0, self.n -1)
            path_pos, target_pos = self.samples[positive]

            # Choose positive sample
            while target_pos != target or positive == index:
                positive = random.randint(0, self.n - 1)
                path_pos, target_pos = self.samples[positive]

            # Choose negative sample
            negative = random.randint(0, self.n - 1)
            path_neg, target_neg = self.samples[negative]
            while target_neg == target:
                negative = random.randint(0, self.n - 1)
                path_neg, target_neg = self.samples[negative]

            query_image = self.loader(path)
            positive_image = self.loader(path_pos)
            negative_image = self.loader(path_neg)

            if self.transform is not None:
                query_image = self.transform(query_image)
                positive_image = self.transform(positive_image)
                negative_image = self.transform(negative_image)


            return (path, query_image, target, positive_image, target_pos, negative_image, target_neg)

        else:

            path, target = self.samples[index]
            image = self.loader(path)
            if self.transform is not None:
                image = self.transform(image)
            return path, image, target


    def __len__(self):
        return len(self.samples)

train_dataset = TinyImageNetData(root='./data/tiny-imagenet-200',
                                 transform=train_transform)

test_dataset = TinyImageNetData(root='./data/tiny-imagenet-200',
                                transform=test_transform,
                                train=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=16)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=16)


# len(test_loader)


embedding_size = 4096
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, embedding_size)
model = model.to(device)

criterion = nn.TripletMarginLoss(margin=1.0)


optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train_loss = [0.3029077, 0.1974788]
print('======================================')
print('.............Training started.........')
print('======================================')
train_loss = []
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (path, query_img, label, pos_img, pos_label, neg_img, neg_label) in enumerate(train_loader):
#         print(i)
        query_img = query_img.to(device)
        pos_img = pos_img.to(device)
        neg_img = neg_img.to(device)
        optimizer.zero_grad()
        query_out = model(query_img)
        pos_out = model(pos_img)
        neg_out = model(neg_img)
        loss = criterion(query_out, pos_out, neg_out)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    if (epoch+1) % 5 == 0:
        curr_lr /= 2
        update_lr(optimizer, curr_lr)
    if (epoch+1) % 3 == 0:
        torch.save(model, 'model_epoch' + str(epoch+1) +'.ckpt')
    loss_epoch = epoch_loss/len(train_loader)
    train_loss.append(loss_epoch)
    print('Mean Loss in epoch ', epoch+1, ' is ', loss_epoch)


torch.save(model.state_dict(), 'params_epoch10.ckpt')
torch.save(model, 'model_epoch10.ckpt')

# XXXXXXXXXX IMPORTANT XXXXXXXXXXXXXX
# From this point, load the model and then work..
# model = torch.load('model_epoch10.ckpt')

# Calculate train feature embeddings
model.eval()
feature_embeddings = torch.zeros([100000, 4096])
# images = []
# Keep track of classes of all 100000 images
classes = []
paths = []
with(torch.no_grad()):
    for i, (path, query_img, label, pos_img, pos_label, neg_img, neg_label) in enumerate(train_loader):
#         print(label)
        query_img = query_img.to(device)
        paths.extend(list(path))
        feature_embeddings[i*16:(i+1)*16,:] = model(query_img)
        if (i+1)% 100 == 0:
            print('Running for Test Image ', i + 1)
        classes.extend(label)


feature_embeddings = feature_embeddings.to(device)

# Calculate testing accuracy
topk = 30
accu = []
done_classes = []

with(torch.no_grad()):
    for i, (path, query_img, lbl) in enumerate(test_loader):

        query_img = query_img.to(device)
        test_embed = model(query_img)
        for (path_i, image_i, lbl_i) in zip(path, test_embed, lbl):

            embed_i = image_i.reshape(1,4096).repeat(100000, 1)
            distance = torch.norm((embed_i - feature_embeddings).double(), 2, 1, True)
            distance = np.squeeze(distance.cpu().numpy())
            # print(distance[:2])
            topk_img = distance.argsort()[:topk]
            accuracy = (np.array(classes)[topk_img] == lbl_i).sum()
            # print(accuracy.item())
            print(topk_img)
            accu.append(accuracy)
            # if lbl_i not in done_classes and len(done_classes) < 5:
            #     done_classes.append(lbl_i)
            #     Image.open(path_i)
            #     top10 = topk_img[:10]
            #     for i in range(10):
            #         Image.open(paths[top10[i]])
            #         print(distance[top10[i]])
            #     bottom10 = distance.argsort()[-10:]
            #     for i in range(10):
            #         Image.open(paths[bottom10[i]])
            #         print(distance[bottom10[i]])
print(np.mean(accu)/ 30)


print('The mean accuracy is ', np.mean(accu)/ 30*100)


# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False,
#                                           num_workers=4)

# Get top 10 bottom 10 image paths
topk = 30
done_classes = []
top_img = []
bottom_img = []
with(torch.no_grad()):
    for i, (path, query_img, lbl) in enumerate(test_loader):
        if i < 2:
            query_img = query_img.to(device)
            test_embed = model(query_img)
            for (path_i, image_i, lbl_i) in zip(path, test_embed, lbl):
                # print(lbl_i)
                embed_i = image_i.reshape(1,4096).repeat(100000, 1)
                distance = torch.norm((embed_i - feature_embeddings).double(), 2, 1, True)
                distance = np.squeeze(distance.cpu().numpy())

                topk_img = distance.argsort()[:topk]

                if lbl_i not in done_classes and len(done_classes) < 5:
                    done_classes.append(lbl_i)
                    top_img.append([path_i, 0])
                    bottom_img.append([path_i, 0])
                    top10 = topk_img[:10]
                    for j in range(10):
                        top_img.append([paths[top10[j]], distance[top10[j]]])
#                         print()
                    bottom10 = distance.argsort()[-10:]
                    for j in range(10):
                        bottom_img.append([paths[bottom10[j]], distance[bottom10[j]]])

# print(np.mean(accu)/ 30)
def showImagesHorizontally(list_of_files, list_of_files2):
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(2,number_of_files,i+1)
        image = imread(list_of_files[i][0])
        imshow(image)
        a.set_title(round(list_of_files[i][1],3), fontsize=10)
        axis('off')
    for j in range(number_of_files):
        a=fig.add_subplot(2,number_of_files,i+j+2)
        image = imread(list_of_files2[j][0])
        imshow(image)
        a.set_title(round(list_of_files2[j][1],3), fontsize=10)
        axis('off')
    # plt.show()

showImagesHorizontally(top_img[44:], bottom_img[44:])


# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True,
#                                            num_workers=4)

# Training precision code, takes toooooooooooo long
topk = 30
train_accu = []

with(torch.no_grad()):
    for i, (path, query_img, label, pos_img, pos_label, neg_img, neg_label) in enumerate(train_loader):

        query_img = query_img.to(device)
        test_embed = model(query_img)
        print('============')
        print(np.mean(train_accu)/30)
        for (path_i, image_i, lbl_i) in zip(path, test_embed, label):
            embed_i = image_i.reshape(1,4096).repeat(100000, 1)
            distance = torch.norm((embed_i - feature_embeddings).double(), 2, 1, True)
            distance = np.squeeze(distance.cpu().numpy())
            topk_img = distance.argsort()[:topk]
            accuracy = (np.array(classes)[topk_img] == lbl_i).sum()

            train_accu.append(accuracy)

print(np.mean(train_accu)/30)
