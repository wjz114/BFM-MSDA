#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, yaml
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
from torch import optim
from torch.utils.data import DataLoader, random_split
from dataloader.dataloader import get_dataset, Cho2017
import matplotlib.pyplot as plt
from utils.setup_utils import (
    get_device,
)
from sklearn.metrics import accuracy_score, cohen_kappa_score
import pandas as pd
import torch.nn.functional as F
# from Modified_EEGNetwork import DG_Network
from model.EEGNet_teacher import get_teacher_model


# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'
'''Argparse'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--subject_num', type=int, default=52)
parser.add_argument('--gpu_num', type=str, default='0')
parser.add_argument('--config_name', type=str, default='bcicompet2a_config') # Replace correct config name here
aargs = parser.parse_args()

# Config setting
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
config_name = aargs.config_name
with open(os.path.join(config_dir, f'{config_name}.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    args = EasyDict(config)
# Set SEED
# seed = args.SEED
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.cuda.manual_seed(seed)


# Set Device
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = aargs.gpu_num
args['device'] = get_device(aargs.gpu_num)
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deterministic = True

# Update configs
args.lr = float(args.lr)
if args.downsampling != 0: args['sampling_rate'] = args.downsampling

train_losses = []
val_accuracies = []
best_val_acc = [0]*args.num_subjects
best_val_loss = [float('inf')]*args.num_subjects
best_epoch = [0]*args.num_subjects

''' Evaluation '''

if args.task == 'BCICompet2a':

    batch_size = args.batch_size

    total_results = []
    total_kappas = []
    labels = []

    best_total_results = []
    best_total_kappas = []
    best_labels = []

    for num_subject in range(args.num_subjects):
        args['target_subject'] = num_subject

        dataset = get_dataset(config_name, args)
        results = np.zeros((dataset.data.shape[0], args.num_classes))

        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False)
        ckpt_path = f'checkpoints/{args.task}_{args.model_name}/{args.model_type}/{args.model_name}_{args.model_type}_lr{args.lr}_batch{args.batch_size}_Epoch{args.EPOCHS}_S{num_subject:02d}.pth'

        print(ckpt_path)

        model = get_teacher_model(args)
        model.load_state_dict(torch.load(ckpt_path))
        model.to(args.device)

        logits = []

        with torch.no_grad():
            for inputs, label in test_dataloader:
                model.eval()
                inputs, label = inputs.to(args.device), label.to(args.device)

                logit = model(inputs)
                logits.append(logit)
                labels.append(label)

                torch.cuda.empty_cache()

            result = torch.cat(logits, dim=0).argmax(axis=1)
            result = F.one_hot(result, num_classes=args.num_classes).detach().cpu().numpy()
            results += result

        results /= 10
        results = results.argmax(axis=1)

        acc_score = accuracy_score(results, dataset.label)
        total_results.append(acc_score)
        total_kappas.append(cohen_kappa_score(results, dataset.label))

    ### Accuracy
    acc_result_df = pd.DataFrame(total_results)
    acc_result_df.index = [f'S{idx + 1}' for idx in range(args.num_subjects)]
    acc_result_df.loc['Avg.'] = acc_result_df.mean()

    ### Kappa
    kappa_result_df = pd.DataFrame(total_kappas)
    kappa_result_df.index = [f'S{idx + 1}' for idx in range(args.num_subjects)]
    kappa_result_df.loc['Avg.'] = kappa_result_df.mean()

    result_df = pd.merge(acc_result_df, kappa_result_df, left_index=True, right_index=True, how='inner')
    result_df.columns = ['Acc.', 'Kappa']

    print('\n\n')
    print('=' * 24)
    print('=' * 7, ' Result ', '=' * 7)
    print(result_df)
    print('=' * 24)
    print('=' * 24)
    print('\n\n')

    result_df.to_csv('BCICompet2a_selection_accuracies-4class.csv', index=True)

elif args.task == 'Cho':

    near_subjects = [2, 28, 44, 30, 40, 42, 14, 4, 45, 17]
    cho_results = []
    cho_kappas = []

    # Load data for near subjects
    alldata = {}
    for num_subject in near_subjects:
        args['target_subject'] = num_subject
        dataset = Cho2017(args.preprocessing, num_subject, args.is_test)
        alldata[num_subject] = dataset

    # Evaluate models on Cho dataset
    for tgt_idx, tgt_subject in enumerate(near_subjects):
        args['target_subject'] = tgt_idx
        dataset = alldata[tgt_subject]
        results = np.zeros((dataset.data.shape[0], args.num_classes))

        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False)
        ckpt_path = f'checkpoints/{args.task}_{args.model_name}/{args.model_type}/{args.model_name}_{args.model_type}_lr{args.lr}_batch{args.batch_size}_Epoch{args.EPOCHS}_S{tgt_idx:02d}.pth'

        print(f"Evaluating Cho dataset for subject {tgt_idx}: {ckpt_path}")
        model = get_teacher_model(args)
        model.load_state_dict(torch.load(ckpt_path))
        model.to(args.device)

        logits = []
        with torch.no_grad():
            for inputs, label in test_dataloader:
                model.eval()
                inputs, label = inputs.to(args.device), label.to(args.device)

                logit = model(inputs)
                logits.append(logit)

                torch.cuda.empty_cache()

            result = torch.cat(logits, dim=0).argmax(axis=1)
            result = F.one_hot(result, num_classes=args.num_classes).detach().cpu().numpy()
            results += result

        results /= 10
        results = results.argmax(axis=1)

        acc_score = accuracy_score(results, dataset.label)
        cho_results.append(acc_score)
        cho_kappas.append(cohen_kappa_score(results, dataset.label))

    cho_acc_df = pd.DataFrame(cho_results, index=[f'S{subject}' for subject in near_subjects], columns=['Acc.'])
    cho_kappa_df = pd.DataFrame(cho_kappas, index=[f'S{subject}' for subject in near_subjects], columns=['Kappa'])
    cho_result_df = pd.concat([cho_acc_df, cho_kappa_df], axis=1)
    cho_result_df.loc['Avg.'] = cho_result_df.mean()

    print('\n\n')
    print('=' * 24)
    print('=' * 7, ' Cho Dataset Result ', '=' * 7)
    print(cho_result_df)
    print('=' * 24)
    print('=' * 24)
    print('\n\n')

    cho_result_df.to_csv('2a_selection_accuracies.csv', index=True)

else :
    raise ValueError("Invalid task specified. Use 'BCICompet2a' or 'Cho'.")


