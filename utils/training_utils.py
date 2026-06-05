import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from scipy.linalg import fractional_matrix_power
import numpy as np
from sklearn.metrics import accuracy_score
import os
import math
import matplotlib.pyplot as plt
from utils.CSutils import CondCSD
from utils.CSutils import CS
from torch.utils.data import DataLoader

def get_criterion():
    
    criterion = torch.nn.CrossEntropyLoss()
    
    return criterion


def get_optimizer(model, args):
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    return optimizer


def get_scheduler(optimizer, args):
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS, eta_min=0)
    
    return scheduler


# def gaussian_kernel(x, y, sigma=2.0):
#     # Flatten the spatial dimensions
#     x_flat = x.view(x.size(0), -1)
#     y_flat = y.view(y.size(0), -1)
#
#     beta = 1. / (2. * sigma ** 2)
#
#     x_square = torch.sum(x_flat ** 2, dim=1, keepdim=True)
#     y_square = torch.sum(y_flat ** 2, dim=1, keepdim=True)
#     xy = torch.matmul(x_flat, y_flat.t())
#     dist = x_square + y_square.t() - 2 * xy
#     return torch.exp(-beta * dist)
#
#
# def mmd_loss(source_features, target_features, kernel=gaussian_kernel):
#     source_kernel = kernel(source_features, source_features)
#     target_kernel = kernel(target_features, target_features)
#     cross_kernel = kernel(source_features, target_features)
#
#     mmd = source_kernel.mean() + target_kernel.mean() - 2 * cross_kernel.mean()
#     return mmd

class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        # Flatten the spatial dimensions
        source = source.view(source.size(0), -1)
        target = target.view(source.size(0), -1)
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX + YY - XY - YX)
        return loss

def print_log(inputs: str):
    print(f'LOG >>> {inputs}')


def get_device(GPU_NUM: str) -> torch.device:
    if torch.cuda.device_count() == 1:
        output = torch.device('cuda')
    elif torch.cuda.device_count() > 1:
        output = torch.device(f'cuda:{GPU_NUM}')
    else:
        output = torch.device('cpu')

    print_log(f'{output} is checked')
    return output

def EA(x):
    """
    Parameters
    ----------
    x : torch.Tensor
        data of shape (batch, 1, num_channels, num_time_samples)

    Returns
    ----------
    XEA : torch.Tensor
        data of shape (batch, 1, num_channels, num_time_samples)
    """
    # Remove the singleton dimension
    x = x.squeeze(1)

    cov = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device=x.device)
    for i in range(x.shape[0]):
        cov[i] = torch.from_numpy(np.cov(x[i].cpu().numpy())).to(x.device)
    refEA = torch.mean(cov, 0)
    sqrtRefEA = torch.from_numpy(fractional_matrix_power(refEA.cpu().numpy(), -0.5)).to(x.device).type(x.dtype)
    XEA = torch.zeros_like(x)
    for i in range(x.shape[0]):
        XEA[i] = torch.matmul(sqrtRefEA, x[i])

    # Add the singleton dimension back
    XEA = XEA.unsqueeze(1)
    return XEA


def combine_train(model, src_data_list, tgt_data, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS, eta_min=0)
    criterion = nn.CrossEntropyLoss()
    beta1 = args.beta1
    warm = args.warm_up
    val_accuracies = []
    loss_values = []
    num_subject = args.target_subject

    for epoch in range(args.EPOCHS):
        model.train()
        running_loss = 0.0
        alpha = math.exp(-epoch + warm) / (1 + math.exp(-epoch + warm))

        for data_batches in zip(*src_data_list, tgt_data):
            src_inputs = []
            src_labels = []
            for src_data in data_batches[:-1]:
                inputs, labels = src_data
                inputs = EA(inputs) # Euclidean Alignment
                src_inputs.append(inputs.to(args.device))
                src_labels.append(labels.to(args.device))

            inputs_tgt, labels_tgt = data_batches[-1]
            inputs_tgt, labels_tgt = inputs_tgt.to(args.device), labels_tgt.to(args.device)
            inputs_tgt = EA(inputs_tgt) # Euclidean Alignment

            combined_inputs = torch.cat(src_inputs, dim=0)
            combined_labels = torch.cat(src_labels, dim=0)
            perm = torch.randperm(combined_inputs.size(0))
            combined_inputs = combined_inputs[perm]
            combined_labels = combined_labels[perm]

            feature_src, output_src = model(combined_inputs, intermediate=True)
            feature_tgt, output_tgt = model(inputs_tgt, intermediate=True)

            marginal_loss = CS(feature_src, feature_tgt)
            cond_loss = CondCSD(feature_src, feature_tgt, output_src, output_tgt)
            cls_loss = criterion(output_src, combined_labels)
            align_loss = marginal_loss + 2 * (1 - alpha) * cond_loss
            loss = cls_loss + beta1 * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(src_data_list[0])
        loss_values.append(running_loss)
        print(f'Epoch {epoch + 1}/{args.EPOCHS}, Loss: {running_loss}')

        scheduler.step()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tgt_data:
                model.eval()
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                labels = torch.argmax(labels, dim=1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        print(f'Accuracy on validation set: {val_acc}%')

    avg_val_acc = sum(val_accuracies[-10:]) / 10
    print(f'Average of last 10 validation accuracies: {avg_val_acc}%')

    save_path = f'checkpoints/{args.task}_{args.model_name}/{args.model_type}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(),
               f'{save_path}/{args.model_name}_{args.model_type}_lr{args.lr}_batch{args.batch_size}_Epoch{args.EPOCHS}_S{num_subject:02d}.pth')

    plt.subplot(2, 1, 1)
    plt.plot(range(1, args.EPOCHS + 1), val_accuracies, label=f'Validation Acc')
    plt.ylim(40, 100)
    plt.title(f'Validation Accuracy for {args.target_subject},beta={args.beta1},lr={args.lr},decay={args.weight_decay}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss_values, label=f'Training Losses')
    plt.title(f'Losses for {args.model_type}, Sigma=5')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.tight_layout()
    plt.show()



    return avg_val_acc


def single_train(model, src_data, tgt_data, args, preload=False,):
    # model = get_teacher_model(args)
    # model.to(args.device)
    # ini_path = f'checkpoints/{args.task}_CS_UDA/ini_EEGNet.pth'
    # model.load_state_dict(torch.load(ini_path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    beta1 = args.beta1
    beta2 = args.beta2
    epsilon = 1e-8  # Small value to prevent division by zero
    loss_values = []  # List to store loss values
    val_accuracies = []
    for epoch in range(args.EPOCHS):
        model.train()
        running_loss = 0.0
        confidence_threshold = 0.9
        # Calculate the alpha value for the epoch
        alpha = math.exp(-epoch + 100) / (1 + math.exp(-epoch + 100))
        # alpha = 1

        for (inputs1, labels1), (inputstgt, labelstgt) in zip(src_data, tgt_data):
            inputs1, labels1 = inputs1.to(args.device), labels1.to(args.device)
            inputs_tgt = inputstgt.to(args.device)

            feature1, output1 = model(inputs1, intermediate=True)
            feature2, output2 = model(inputs_tgt, intermediate=True)
            # feature1 = feature1 - feature1.mean(dim=0)
            # feature2 = feature2 - feature2.mean(dim=0)

            # Calculate pseudo-labels for target data
            # pseudo_labels_tgt = torch.argmax(output2, dim=1)
            # confidence_scores = torch.max(F.softmax(output2, dim=1), dim=1)[0]

            # Apply confidence thresholding
            # mask = confidence_scores >= confidence_threshold
            # high_confidence_labels = pseudo_labels_tgt[mask]
            # high_confidence_features = feature2[mask]

            # if high_confidence_labels.size(0) > 0:
                # high_confidence_labels = high_confidence_labels.view(-1, 1)
                # output1 = output1.view(-1, 1)
                # cond_loss = CondCSD(feature1[mask], feature2[mask], output1[mask], pseudo_labels_tgt[mask])
                # print(f'Cond Loss: {cond_loss}')
            # else:
            #     cond_loss = torch.tensor(0.0, device=args.device)

            marginal_loss = CS(feature1, feature2)
            # marginal_loss = calculate_mmd(feature1, feature2)
            cond_loss = CondCSD(feature1, feature2, output1, output2)
            cls_loss = criterion(output1, labels1)
            align_loss = marginal_loss + 2*(1-alpha) * cond_loss
            loss = cls_loss + beta1 * align_loss

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        running_loss /= len(src_data)
        loss_values.append(running_loss)  # Store the average loss for the epoch
        print(f'Epoch {epoch + 1}/{args.EPOCHS}, Loss: {running_loss}')

        # Validation
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tgt_data:
                model.eval()
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                labels = torch.argmax(labels, dim=1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        print(f'Accuracy on target set: {val_acc}%')

    # Calculate the average of the last 10 validation accuracies
    avg_val_acc = sum(val_accuracies[-10:]) / 10
    print(f'Average of last 10 validation accuracies: {avg_val_acc}%')

    # Save  model
    save_path = f'checkpoints/{args.task}_{args.model_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(),
               f'{save_path}/{args.model_name}_{args.model_type}_lr{args.lr}_batch{args.batch_size}_Epoch{args.EPOCHS}_src0{args.src_subject}_tgt0{args.tgt_subject}.pth')

    # Plot the training loss vs epoch
    plt.subplot(2, 1, 1)
    plt.plot(range(1, args.EPOCHS + 1), val_accuracies, label=f'Validation Acc')
    plt.ylim(10, 100)
    plt.title(f'Validation Accuracy for {args.model_type},beta={args.beta1},src={args.src_subject},tgt={args.tgt_subject}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss_values, label=f'Training Losses')
    plt.title(f'Losses for {args.model_type}, src={args.src_subject},tgt={args.tgt_subject}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.tight_layout()
    plt.show()

    torch.cuda.empty_cache()
    return avg_val_acc

def multi_source(model, src_datasets, tgt_dataset, args):
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS, eta_min=0)

    criterion = nn.CrossEntropyLoss()
    beta1 = args.beta1
    warm = args.warm_up
    val_accuracies = []
    loss_values = []
    num_subject = args.target_subject

    # Create DataLoaders for source and target datasets
    src_dataloaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=True) for dataset in src_datasets]
    tgt_dataloader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.EPOCHS):
        model.train()
        running_loss = 0.0
        alpha = math.exp(-epoch + warm) / (1 + math.exp(-epoch + warm))

        for data_batches in zip(*src_dataloaders, tgt_dataloader):
            src_inputs = []
            src_labels = []
            for src_data in data_batches[:-1]:
                inputs, labels = src_data
                src_inputs.append(inputs.to(args.device))
                src_labels.append(labels.to(args.device))

            inputs_tgt, labels_tgt = data_batches[-1]
            inputs_tgt, labels_tgt = inputs_tgt.to(args.device), labels_tgt.to(args.device)

            features = []
            outputs = []
            for src_input in src_inputs:
                feature, output = model(src_input, intermediate=True)
                features.append(feature - feature.mean(dim=0))
                outputs.append(output)

            feature_tgt, output_tgt = model(inputs_tgt, intermediate=True)
            feature_tgt = feature_tgt - feature_tgt.mean(dim=0)

            cs_losses = [CS(f, feature_tgt) for f in features]
            cond_losses = [CondCSD(f, feature_tgt, o, output_tgt) for f, o in zip(features, outputs)]

            cs_losses_src = [CS(f1, f2) for i, f1 in enumerate(features) for f2 in features[i+1:]]
            cond_losses_src = [CondCSD(f1, f2, o1, o2) for i, (f1, o1) in enumerate(zip(features, outputs)) for f2, o2 in zip(features[i+1:], outputs[i+1:])]

            marginal_st = sum(cs_losses) / len(cs_losses)
            marginal_ss = sum(cs_losses_src) / len(cs_losses_src)
            marginal_loss = marginal_st + marginal_ss
            cond_st = sum(cond_losses) / len(cond_losses)
            cond_ss = sum(cond_losses_src) / len(cond_losses_src)
            cond_loss = cond_st + cond_ss

            cls_losses = [criterion(output, label) for output, label in zip(outputs, src_labels)]
            cls_loss = sum(cls_losses) / len(cls_losses)
            align_loss = marginal_loss +  (1 - alpha) * cond_loss
            loss = cls_loss + beta1 * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(src_dataloaders[0])
        loss_values.append(running_loss)
        print(f'Epoch {epoch + 1}/{args.EPOCHS}, Loss: {running_loss}')

        scheduler.step()

        # Validation
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tgt_dataloader:
                model.eval()
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                labels = torch.argmax(labels, dim=1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        print(f'Accuracy on validation set: {val_acc}%')

    avg_val_acc = sum(val_accuracies[-10:]) / 10
    print(f'Average of last 10 validation accuracies: {avg_val_acc}%')

    save_path = f'checkpoints/{args.task}_{args.model_name}/{args.model_type}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(),
               f'{save_path}/{args.model_name}_{args.model_type}_lr{args.lr}_batch{args.batch_size}_Epoch{args.EPOCHS}_S{num_subject:02d}.pth')

    plt.subplot(2, 1, 1)
    plt.plot(range(1, args.EPOCHS + 1), val_accuracies, label=f'Validation Acc')
    plt.ylim(40, 100)
    plt.title(f'Validation Accuracy for {args.target_subject},beta={args.beta1},lr={args.lr},decay={args.weight_decay}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss_values, label=f'Training Losses')
    plt.title(f'Losses for {args.model_type}, Sigma=5')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.tight_layout()
    plt.show()

    return avg_val_acc