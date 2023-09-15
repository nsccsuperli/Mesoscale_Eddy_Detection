from PIL import Image
import torch
from torch.utils.data import Dataset
import csv
import torch
import imageio
# import tensorflow as tf
import os
from torchvision import transforms
# from medpy.io import load
# from albumentations import *
import numpy as np
from PIL import Image



class MyDataSet(Dataset):
    """自定义数据集"""
    # 2023616 添加SST数据集
    # def __init__(self, root,transform):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        # edge_train_data = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/edge_train_data.npy'), 3)[:,:,:,0]
        # SSH_train = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/train_img_data.npy'),3)
        # SSH_train_bak = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/train_img_data_bak.npy'),3)
        #
        #
        # Seg_train = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/seg_train_data.npy'), 3)

        #  使用新的数据集
        edge_train_data = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/train_seg_edge_data.npy'), 3)
        SSH_train = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/train_ssh_data.npy'),3)
        SST_train = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/train_sst_data.npy'),3)
        SSH_train_bak = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/train_img_data_bak.npy'),3)
        Seg_train = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/train_seg_eddy_data.npy'), 3)

        seg_train = np.eye(3)[Seg_train[:, :, :, 0]]
        edge2 = np.eye(3)[edge_train_data[:, :, :, 0]]
        # self.input1 = torch.tensor(SSH_train)
        # self.SSH_train_bak = torch.tensor(SSH_train_bak)
        # self.edge1 = torch.tensor(edge_train_data)
        # self.mask1 = torch.tensor(seg_train)
        self.input = SSH_train
        self.input2 = SST_train
        # self.SSH_train_bak = SSH_train_bak
        self.edge = edge_train_data
        self.mask = seg_train
        self.edge2 = edge2
        # self.images_path = torch.tensor(np.load(os.path.join(data, "train.npy")))
        # self.images_class = torch.tensor(np.load(os.path.join(label, "label.npy")))
        self.transform = transform

    def __len__(self):
        return self.input.shape[0]  # 返回数据的总个数

    def __getitem__(self, index):
        # 计算均值和标准差
        # self.input[self.input > 9] = 0
        # mean = torch.zeros(1)
        # std = torch.zeros(1)
        # mean += self.input[:, :, :,0].mean()
        # std += self.input[:, :, :,0].std()
        # mean.div_(self.input.shape[0])
        # std.div_(self.input.shape[0])
        # mean_1,std_1 = list(mean.numpy()), list(std.numpy())

        input = self.input[index, :, :]  # 读取每一个npy的数据
        input2 = self.input2[index, :, :]  # 读取每一个npy的数据
        edge = self.edge[index,:,:]  # 读取每一个npy的数据
        mask = self.mask[index,:,:]  # 读取每一个npy的数据
        edge2 = self.edge2[index,:,:]  # 读取每一个npy的数据



        # input = np.expand_dims(input, axis=0)
        # input = torch.Tensor(input)
        # input = torch.cat([input, input, input], dim=0)
        ###############################################################################################################3


        # temp = input[:, :, 0]

        # train_input = self.SSH_train_bak[index,:, :]
        # # train_input = train_input[index,:, :, 0]
        # print(input[:, :, 0])
        # totensor_tool = transforms.ToTensor()
        # img_tensor = totensor_tool(input)
        # norma_tool =transforms.Normalize((0.5,), (0.5,))
        # img_nor = norma_tool(img_tensor)
        # print(img_tensor)
        #
        #
        # img_tensor_train = totensor_tool(train_input)
        # norma_tool =transforms.Normalize((0.5,), (0.5,))
        # img_nor_train = norma_tool(img_tensor_train)
        # print(img_tensor)
        ###############################################################################################################3
        # mask = mask.type(torch.long)
#       针对不做数据增强的处理方式
#         input[input > 9] = 0
        # input = torch.tensor(input)
        # input = torch.transpose(torch.transpose(input, 0, 2), 1, 2)
        # norma_tool =transforms.Normalize((0.001,), (0.00001,))
        # input = norma_tool(input)

        if self.transform is not None:
            input = self.transform(input)
            input2 = self.transform(input2)
            # train_input = self.transform(train_input)
            # temp1 = input[0, :,:]
            # train_input = train_input[0, :,:]
            # mask1 = mask[:, :, 0]
            # edge1 = edge

        return input,input2, mask,edge,edge2  # 返回数据还有标签



class valDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        # edge_test_data = np.load('/home/eddy/GSCNN-master/datasets/edge_test_data.npy')
        # # datalength\heigh weith channel
        # SSH_test = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/test_img_data.npy'), 3)
        # Seg_test = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/seg_test_data.npy'), 3)
        # 使用新的数据
        edge_test_data = np.load('/home/eddy/GSCNN-master/datasets/val_seg_edge_data.npy')
        # datalength\heigh weith channel
        SSH_test = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/val_ssh_data.npy'), 3)
        SST_test = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/val_sst_data.npy'), 3)
        Seg_test = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/val_seg_eddy_data.npy'), 3)

        seg_test = np.eye(3)[Seg_test[:, :, :, 0]]

        self.input = SSH_test
        self.input2 = SST_test
        self.mask = seg_test
        self.edge = edge_test_data
        # self.images_path = torch.tensor(np.load(os.path.join(data, "val.npy")))
        # self.images_class = torch.tensor(np.load(os.path.join(label, "val_label.npy")))
        # self.input1 = torch.tensor(SSH_test)
        # self.mask1 = torch.tensor(seg_test)
        # self.edge1 = torch.tensor(edge_test_data)
        self.transform = transform


    def __len__(self):
        return self.input.shape[0]  # 返回数据的总个数

    def __getitem__(self, index):
        input = self.input[index, :, :]  # 读取每一个npy的数据
        input2 = self.input2[index, :, :]  # 读取每一个npy的数据
        edge = self.edge[index, :, :]  # 读取每一个npy的数据
        mask = self.mask[index,:,:,:]  # 读取每一个npy的数据

        # mask = Image.fromarray(mask)
        # edge = Image.fromarray(edge)

        # input = np.expand_dims(input, axis=0)
        # input = torch.Tensor(input)
        # img = torch.cat([img, img, img], dim=0)
        ###############################################################################################################3
        # print(input[:, :, 0])
        # totensor_tool = transforms.ToTensor()
        # img_tensor = totensor_tool(input)
        # norma_tool =transforms.Normalize((0.5,), (0.5,))
        # img_nor = norma_tool(img_tensor)
        # print(img_tensor)
        ###############################################################################################################3
        ###############################################################################################################3
        # mask = mask.type(torch.long)
        # 不做数据增强的处理方式
        # input[input > 9] = 0
        # input = torch.tensor(input)
        # input = torch.transpose(torch.transpose(input, 0, 2), 1, 2)
        # norma_tool =transforms.Normalize((0.001,), (0.00001,))
        # input = norma_tool(input)
        # 做数据增强的方式
        if self.transform is not None:
            input = self.transform(input)
            input2 = self.transform(input2)
        return input,input2, mask ,edge  # 返回数据还有标签

    #
    # def collate_fn(batch):
    #     # 官方实现的default_collate可以参考
    #     # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
    #     images, labels = tuple(zip(*batch))
    #
    #     images = torch.stack(images, dim=0)
    #     labels = torch.as_tensor(labels)
    #     return images, labels