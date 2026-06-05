"""
Class activation topography (CAT) for EEG model visualization, combining class activity map and topography
Code: Class activation map (CAM) and then CAT

refer to high-star repo on github: 
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/grad_cam
AND
https://github.com/eeyhsong/EEG-Conformer

"""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
from easydict import EasyDict
import torch
import os, yaml
# from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import math
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
from torch.backends import cudnn
from utils.visual_utils import GradCAM
from utils.setup_utils import get_device
# from model.MSN_student import get_student_model
from model.EEGNet_teacher import get_teacher_model
from dataloader.dataloader import get_dataset
# from torchsummary import summary
cudnn.benchmark = False
cudnn.deterministic = True
import random


'''Argparse'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='bcicompet2a_config')
aargs = parser.parse_args()

# Config setting
with open(r'E:\PycharmProjects\BFM-MSDA\configs\bcicompet2a_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    args = EasyDict(config)
config_name = aargs.config_name
# Set SEED
seed = args.SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

# Set Device
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_NUM
args['device'] = get_device(args.GPU_NUM)
device = torch.device("cpu")
# Update configs
if args.downsampling != 0: args['sampling_rate'] = args.downsampling

# load data
all_train_data = {}

for subject in range(args.num_subjects):
    args['target_subject'] = subject

    all_train_data[subject] = get_dataset(config_name, args)


# ! A crucial step for adaptation on Transformer
# reshape_transform  b 61 40 -> b 40 1 61
def reshape_transform(tensor):
    # result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return tensor

for subject in range(args.num_subjects):
    # model initialization
    model = get_teacher_model(args)
    student = get_teacher_model(args)
    # model = Conformer()
    args['target_subject'] = subject

    ckpt_path = f'checkpoints/{args.task}_{args.model_name}/{args.model_type}/{args.model_name}_{args.model_type}_lr{args.lr}_batch{args.batch_size}_Epoch{args.EPOCHS}_S{subject:02d}.pth'
    student_path = f'checkpoints/{args.task}_Baseline/LOSO/Baseline_LOSO_lr{args.lr}_batch{args.batch_size}_Epoch{args.EPOCHS}_S{subject:02d}.pth'
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)
    student.load_state_dict(torch.load(student_path))
    student.to(device)

    target_layers = list(model.backbone.flatten.children())  # set the target layer
    student_layers = list(student.backbone.flatten.children())

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_transform)
    cam_student = GradCAM(model=student, target_layers=student_layers, use_cuda=False, reshape_transform=reshape_transform)

    # dataset division
    dataset = all_train_data[subject]  # 288, 1, 22, 751 BCIcompetition IV 2a
    data = dataset.data  # ndarray
    label = dataset.label  # ndarray
    target_category = label
    left = np.where(label == 0)
    data_left = data[left]
    target_category_left = label[left]  # set the class (class activation mapping)

    right = np.where(label == 1)
    data_right = data[right]
    target_category_right = label[right]

    # model.to(args.device)
    # summary(model, (1, 22, 751))

    # # used for cnn model without transformer
    # model.load_state_dict(torch.load('./model/model_cnn.pth', map_location=device))
    # target_layers = [model[0].projection]  # set the layer you want to visualize, you can use torchsummary here to find the layer index
    # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)



    import mne

    biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]  # for bci competition iv 2a
    biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
    biosemi_montage.dig = [biosemi_montage.dig[i+3] for i in index]
    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=200., ch_types='eeg')

    ''' All Data'''
    # all_cam = []
    # student_cam = []
    # # this loop is used to obtain the cam of each trial/sample
    # for i in range(288):
    #     test = torch.as_tensor(data[i:i + 1, :, :, :], dtype=torch.float32)
    #
    #     test = torch.autograd.Variable(test, requires_grad=True)
    #     print(test.shape)
    #     grayscale_cam = cam(input_tensor=test, target_category=int(target_category[i]))
    #     grayscale_cam = grayscale_cam[0, :]
    #     all_cam.append(grayscale_cam)
    #
    # for i in range(288):
    #     test = torch.as_tensor(data[i:i + 1, :, :, :], dtype=torch.float32)
    #
    #     test = torch.autograd.Variable(test, requires_grad=True)
    #     print(test.shape)
    #     grayscale_cam = cam_student(input_tensor=test, target_category=int(target_category[i]))
    #     grayscale_cam = grayscale_cam[0, :]
    #     student_cam.append(grayscale_cam)
    #
    # # the mean of all data
    # all_data = np.squeeze(np.mean(data, axis=0))
    # all_data = (all_data - np.mean(all_data)) / np.std(all_data)
    # mean_all = np.mean(all_data, axis=1)
    #
    # # the mean of all cam for teacher
    # all_cam = np.mean(all_cam, axis=0)
    # all_cam = (all_cam - np.mean(all_cam)) / np.std(all_cam)
    # mean_all_cam = np.mean(all_cam, axis=1)
    # mean_all_cam = (mean_all_cam - np.mean(mean_all_cam)) / np.std(mean_all_cam)
    #
    # # the mean of all cam for student
    # student_cam = np.mean(student_cam, axis=0)
    # student_cam = (student_cam - np.mean(student_cam)) / np.std(student_cam)
    # mean_student_cam = np.mean(student_cam, axis=1)
    # mean_student_cam = (mean_student_cam - np.mean(mean_student_cam)) / np.std(mean_student_cam)
    #
    # # apply cam on the input data
    # hyb_all = all_data * all_cam
    # hyb_all = (hyb_all - np.mean(hyb_all)) / np.std(hyb_all)
    # mean_hyb_all = np.mean(hyb_all, axis=1)
    #
    # # mean_hyb_all = mean_all_cam * mean_all
    # evoked = mne.EvokedArray(all_data, info)
    # evoked.set_montage(biosemi_montage)
    #
    ''' Left Data'''
    all_cam_left = []
    # this loop is used to obtain the cam of each trial/sample
    for i in range(36):
        test = torch.as_tensor(data_left[i:i+1, :, :, :], dtype=torch.float32)

        test = torch.autograd.Variable(test, requires_grad=True)
        print(test.shape)
        grayscale_cam = cam(input_tensor=test, target_category=int(target_category_left[i]))
        grayscale_cam = grayscale_cam[0, :]
        all_cam_left.append(grayscale_cam)

    # the mean of all data
    left_all_data = np.squeeze(np.mean(data_left, axis=0))
    left_all_data = (left_all_data - np.mean(left_all_data)) / np.std(left_all_data)
    mean_all_left = np.mean(left_all_data, axis=1)

    # the mean of all ca
    left_all_cam = np.mean(all_cam_left, axis=0)
    left_all_cam = (left_all_cam - np.mean(left_all_cam)) / np.std(left_all_cam)
    mean_all_cam_left = np.mean(left_all_cam, axis=1)
    # mean_all_cam_left = (mean_all_cam_left - np.mean(mean_all_cam_left)) / np.std(mean_all_cam_left)

    # apply cam on the input data
    hyb_all_left = left_all_data * left_all_cam
    hyb_all_left = (hyb_all_left - np.mean(hyb_all_left)) / np.std(hyb_all_left)
    mean_hyb_all_left = np.mean(hyb_all_left, axis=1)
    mean_hyb_all_left = mean_all_cam_left * mean_all_left

    evoked_left = mne.EvokedArray(left_all_data, info)
    evoked_left.set_montage(biosemi_montage)

    ''' Right Data'''
    all_cam_right = []
    # this loop is used to obtain the cam of each trial/sample
    for i in range(36):
        test = torch.as_tensor(data_right[i:i+1, :, :, :], dtype=torch.float32)

        test = torch.autograd.Variable(test, requires_grad=True)
        print(test.shape)
        grayscale_cam = cam(input_tensor=test, target_category=int(target_category_right[i]))
        grayscale_cam = grayscale_cam[0, :]
        all_cam_right.append(grayscale_cam)
    # the mean of all data
    right_all_data = np.squeeze(np.mean(data_right, axis=0))
    right_all_data = (right_all_data - np.mean(right_all_data)) / np.std(right_all_data)
    mean_all_right = np.mean(right_all_data, axis=1)

    # the mean of all ca
    right_all_cam = np.mean(all_cam_right, axis=0)
    right_all_cam = (right_all_cam - np.mean(right_all_cam)) / np.std(right_all_cam)
    mean_all_cam_right = np.mean(right_all_cam, axis=1)
    # mean_all_cam_right = (mean_all_cam_right - np.mean(mean_all_cam_right)) / np.std(mean_all_cam_right)

    # apply cam on the input data
    hyb_all_right = right_all_data * right_all_cam
    hyb_all_right = (hyb_all_right - np.mean(hyb_all_right)) / np.std(hyb_all_right)
    mean_hyb_all_right = np.mean(hyb_all_right, axis=1)
    mean_hyb_all_right = mean_all_cam_right * mean_all_right

    evoked_right = mne.EvokedArray(right_all_data, info)
    evoked_right.set_montage(biosemi_montage)

    '''Plot 2x2 Subplot'''
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

    # print(mean_all_test)
    plt.subplot(2,2,1)
    im1, cn1 = mne.viz.plot_topomap(mean_all_left, evoked_left.info, show=False, axes=ax[0][0], res=1200)
    for line in ax[0][0].lines:
        line.set_color('black')
        line.set_linewidth(1)

    plt.subplot(2,2,2)
    im2, cn2 = mne.viz.plot_topomap(mean_hyb_all_left, evoked_left.info, show=False, axes=ax[1][0], res=1200)
    for line in ax[1][0].lines:
        line.set_color('black')
        line.set_linewidth(1)

    ax[0,0].set_title('Left', fontweight='bold')

    plt.subplot(2,2,3)
    im3, cn3 = mne.viz.plot_topomap(mean_all_right, evoked_right.info, show=False, axes=ax[0][1], res=1200)
    for line in ax[0][1].lines:
        line.set_color('black')
        line.set_linewidth(1)

    ax[0,1].set_title('Right', fontweight='bold')

    plt.subplot(2,2,4)
    im4, cn4 = mne.viz.plot_topomap(mean_hyb_all_right, evoked_right.info, show=False, axes=ax[1][1], res=1200)
    for line in ax[1][1].lines:
        line.set_color('black')
        line.set_linewidth(1)

    plt.colorbar(im3, ticks=[0])
    plt.colorbar(im4, ticks=[0])
    fig.tight_layout()
    plt.savefig(f'./topography_Rawandteacher_{subject}.png')
    plt.show()
    #
    # '''Plot all topo'''
    # fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    # # print(mean_all_test)
    # plt.subplot(3,1,1)
    # im1, cn1 = mne.viz.plot_topomap(mean_all, evoked.info, show=False, axes=ax[0], res=2000)
    # for line in ax[0].lines:
    #     line.set_color('black')
    #     line.set_linewidth(1)
    #
    # plt.subplot(3,1,2)
    # im2, cn2 = mne.viz.plot_topomap(mean_all_cam, evoked.info, show=False, axes=ax[1], res=2000)
    # for line in ax[1].lines:
    #     line.set_color('black')
    #     line.set_linewidth(1)
    #
    # plt.subplot(3,1,3)
    # im3, cn3 = mne.viz.plot_topomap(mean_student_cam, evoked.info, show=False, axes=ax[2], res=2000)
    # for line in ax[2].lines:
    #     line.set_color('black')
    #     line.set_linewidth(1)
    # torch.cuda.empty_cache()
    # ax[0].set_title(f'S{subject+1}', fontweight='bold')
    # # plt.colorbar(im1, ticks=[0])
    # # plt.colorbar(im2, ticks=[0])
    # # plt.colorbar(im3, ticks=[0])
    #
    # # fig.tight_layout()
    # plt.savefig(f'./topography_raw_vs_teacher_vs_student_{subject}.png')
    # plt.show()
    #
