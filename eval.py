import json
import os
import shutil
from time import time

import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet34, VGG16,NetC_MNIST
from networks.models import Denormalizer, Normalizer
from torch import nn
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from DWT.DWT import noise_trigger,DWT,merge_N_HH,merge_gray_HH,noise_trigger_gray,merge_N_HL,merge_gray_HL

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10":
        netC = VGG16().to(opt.device)
    if opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "imagenet":
        netC = ResNet34().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def eval(
    netC,
    optimizerC,
    schedulerC,
    identity_grid,
    test_dl,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_ae_loss = 0
    nomalizer = Normalizer(opt)
    # criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            inputs_bd = inputs
            for i in range(bs):
                inputs_bd[i] = merge_gray_HH(inputs_bd[i].unsqueeze(0), identity_grid.unsqueeze(0))
                inputs_bd[i] = nomalizer(inputs_bd[i]).to(opt.device)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample
            info_string = "Clean Acc: {:.4f}| Bd Acc: {:.4f}".format(
                acc_clean,  acc_bd
                )
            progress_bar(batch_idx, len(test_dl), info_string)


def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "imagenet":
        opt.num_classes = 10
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "imagenet":
        opt.input_height = 128
        opt.input_width = 128
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    test_dl = get_dataloader(opt, False)
    identity_image = noise_trigger_gray(opt.input_height)
    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")

    if os.path.exists(opt.ckpt_path):
        state_dict = torch.load(opt.ckpt_path)
        netC.load_state_dict(state_dict["netC"])
    else:
        print("Pretrained model doesnt exist")
        exit()

    eval(
        netC,
        optimizerC,
        schedulerC,
        identity_image,
        test_dl,
        opt,
    )


if __name__ == "__main__":
    main()
