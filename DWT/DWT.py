# coding:utf-8
import torch.nn as nn
import config
from config import *
import torch
import os, cv2, torchvision
from PIL import Image
import numpy as np
from torchvision import transforms as trans
import torch
import numpy
from torch.nn import functional as F
import math
import torch
LL_b = 0.4
HL_b = 0.4
HH_b =  0.4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    # print(x_LL.shape)
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width]) #[1, 12, 56, 56]
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    # print(out_batch, out_channel, out_height, out_width) #1 3 112 112
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    # print(x1.shape) #torch.Size([1, 3, 56, 56])
    # print(x2.shape) #torch.Size([1, 3, 56, 56])
    # print(x3.shape) #torch.Size([1, 3, 56, 56])
    # print(x4.shape) #torch.Size([1, 3, 56, 56])
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)



class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

##rotation
def rotation(img_torch,angle):
    angle = -angle * math.pi / 180
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0],
        [math.sin(angle), math.cos(angle), 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size(), align_corners=False)
    output = F.grid_sample(img_torch.unsqueeze(0), grid, align_corners=False)
    new_img_torch = output[0]
    return new_img_torch

def noise_trigger(input_height):
    tigger_image2 = Image.open('./DWT/3.JPEG')
    transform = trans.Compose([
        trans.Resize((input_height, input_height)),
        trans.ToTensor()
    ])
    tigger_image2 = transform(tigger_image2)

    return tigger_image2

def noise_trigger_gray(input_height):
    tigger_image2 = Image.open('./DWT/3.JPEG')
    transform = trans.Compose([
        trans.Resize((input_height, input_height)),
        trans.Grayscale(num_output_channels=1),
        trans.ToTensor()
    ])
    tigger_image2 = transform(tigger_image2)
    torchvision.utils.save_image(tigger_image2, './back_tigger_HH.jpg')
    return tigger_image2

def merge_gray_HL(image1, tigger_image2):
    torchvision.utils.save_image(image1, './back_tigger-1.jpg')
    image1 = image1 * 255
    tigger_image2=tigger_image2*255
    dwt = DWT()
    tigger_image2=dwt(tigger_image2)
    tigger_image2.to(device)
    image1 = dwt(image1).to(device)
    back_img = torch.zeros([1, 4, image1.size(2), image1.size(2)]).float().to(device)
    # LL
    back_img[:, 0, :] = image1[:, 0, :]
    HL_tigger1 = rotation(tigger_image2[:, 1, :], -5)
    HL_tigger2 = rotation(tigger_image2[:, 1, :], -10)
    HL_tigger3 = rotation(tigger_image2[:, 1, :], 5)
    HL_tigger4 = rotation(tigger_image2[:, 1, :], 10)
    tigger_image2[:, 1, :] = torch.mean(torch.stack([HL_tigger1, HL_tigger2, HL_tigger3, HL_tigger4]),
                                                       0).unsqueeze(0)
    back_img[:, 1, :] =torch.add((1 - HL_b) * image1[:, 1, :].squeeze(0) ,HL_b * tigger_image2[:,1,:].to(device).squeeze(0)) .unsqueeze(0)
    # # LH
    back_img[:, 2, :] = image1[:, 2, :]
    # HH
    back_img[:, 3, :] = image1[:, 3, :]
    iwt = IWT()
    back_img_tensor = iwt(back_img)
    back_img_tensor = back_img_tensor / 255
    torchvision.utils.save_image(back_img_tensor, './back_tigger-2.jpg')
    back_img_tensor = torch.clamp(back_img_tensor, 0, 0.99)
    return back_img_tensor
def merge_gray_HH(image1, tigger_image2):
    torchvision.utils.save_image(image1, './back_tigger-1.jpg')
    image1 = image1 * 255
    tigger_image2=tigger_image2*255
    dwt = DWT()
    tigger_image2 = dwt(tigger_image2)
    tigger_image2.to(device)
    image1 = dwt(image1).to(device)
    back_img = torch.zeros([1, 4, image1.size(2), image1.size(2)]).float().to(device)
    # LL
    back_img[:, 0, :] = image1[:, 0, :]
    back_img[:, 1, :] = image1[:, 1, :]
    #LH
    back_img[:, 2, :] = image1[:, 2, :]
    # HH
    back_img[:, 3, :] = torch.add((1 - HL_b) * image1[:, 3, :].squeeze(0),
                                  HL_b * tigger_image2[:, 3, :].to(device).squeeze(0)).unsqueeze(0)
    iwt = IWT()
    back_img_tensor = iwt(back_img)
    back_img_tensor = back_img_tensor / 255
    back_img_tensor = torch.clamp(back_img_tensor, 0, 0.99)
    torchvision.utils.save_image(back_img_tensor, './back_tigger-2.jpg')
    # torchvision.utils.save_image(back_img_tensor, './back_tigger_HH.png')
    return back_img_tensor
def merge_N_HH(image1, tigger_image2):
    torchvision.utils.save_image(image1, './back_tigger-1.jpg')
    image1 = image1 * 255
    tigger_image2=tigger_image2*255
    dwt = DWT()
    tigger_image2=dwt(tigger_image2)
    tigger_image2.to(device)
    image1 = dwt(image1).to(device)
    back_img = torch.zeros([1, 12, image1.size(2), image1.size(2)]).float().to(device)
    back_img[:, 0 * 3:0 * 3 + 3:, :] = image1[:, 0 * 3:0 * 3 + 3:, :]
    back_img[:, 2 * 3:2 * 3 + 3:, :] = image1[:, 2 * 3:2 * 3 + 3:, :]
    # HH
    back_img[:,  1* 3:1 * 3 + 3:, :] = image1[:, 1 * 3:1 * 3 + 3:, :]
    back_img[:, 3 * 3:3 * 3 + 3:, :] = torch.add((1 - HL_b) * image1[:, 9:12:, :].squeeze(0),
                                                 HL_b * tigger_image2[:, 9:12:, :].to(device).squeeze(0)).unsqueeze(0)
    iwt = IWT()
    back_img_tensor = iwt(back_img)
    back_img_tensor=back_img_tensor/255
    back_img_tensor = torch.clamp(back_img_tensor, 0, 0.99)
    torchvision.utils.save_image(back_img_tensor, './back_tigger-2.jpg')
    return back_img_tensor
def merge_N_HL(image1, tigger_image2):
    torchvision.utils.save_image(image1, './back_tigger-1.jpg')
    image1 = image1 * 255
    tigger_image2=tigger_image2*255
    dwt = DWT()
    tigger_image2=dwt(tigger_image2)
    tigger_image2.to(device)
    image1 = dwt(image1).to(device)
    back_img = torch.zeros([1, 12, image1.size(2), image1.size(2)]).float().to(device)
    # LL
    back_img[:, 0: 3:, :] = image1[:, 0: 3:, :]
    HL_tigger1 = rotation(tigger_image2[:, 1 * 3:1 * 3 + 3:, :].squeeze(0), -5)
    HL_tigger2 = rotation(tigger_image2[:, 1 * 3:1 * 3 + 3:, :].squeeze(0), -10)
    HL_tigger3 = rotation(tigger_image2[:, 1 * 3:1 * 3 + 3:, :].squeeze(0), 5)
    HL_tigger4 = rotation(tigger_image2[:, 1 * 3:1 * 3 + 3:, :].squeeze(0), 10)
    tigger_image2[:, 1 * 3:1 * 3 + 3:, :] = torch.mean(torch.stack([HL_tigger1, HL_tigger2, HL_tigger3, HL_tigger4]),
                                                       0).unsqueeze(0)
    back_img[:, 3:6:, :] =torch.add((1 - HL_b) * image1[:, 3:6:, :].squeeze(0) , HL_b * tigger_image2[:,3:6:,:].to(device).squeeze(0)) .unsqueeze(0)
    # LH
    back_img[:, 2 * 3:2 * 3 + 3:, :] = image1[:, 2 * 3:2 * 3 + 3:, :]
    # HH
    back_img[:, 3 * 3:3 * 3 + 3:, :] = image1[:, 3 * 3:3 * 3 + 3:, :]
    iwt = IWT()
    back_img_tensor = iwt(back_img)
    back_img_tensor = back_img_tensor / 255
    back_img_tensor = torch.clamp(back_img_tensor, 0, 0.99)
    torchvision.utils.save_image(back_img_tensor, './back_tigger-2.jpg')
    return back_img_tensor





