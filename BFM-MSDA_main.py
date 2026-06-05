import os, yaml
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.optim as optim
from model.EEGNet_teacher import get_teacher_model
from utils.CSutils import CondCSD
from utils.CSutils import CS
import numpy as np
from utils.setup_utils import (
    get_device,
)
from utils.training_utils import EA
import random
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader, random_split
from dataloader.dataloader import get_dataset, Cho2017
import math
import torch.nn.functional as F
import pandas as pd
import time

def compute_weights(features, feature_tgt):
    """Compute weights using softmax on negative distances."""
    distances = [CS(f, feature_tgt) for f in features]
    neg_distances = [-d for d in distances]  # Negate distances
    weights = F.softmax(torch.tensor(neg_distances), dim=0)  # Apply softmax
    return weights

# Training function
def multi_train(src_data_list, tgt_data, args):
    model = get_teacher_model(args)
    model.to(args.device)
    # ini_path = f'checkpoints/{args.task}_CS_UDA/ini_EEGNet.pth'
    # model.load_state_dict(torch.load(ini_path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS, eta_min=0)

    criterion = nn.CrossEntropyLoss()
    beta = args.beta
    warm = args.warm_up
    alpha = args.alpha
    # beta2 = args.beta2
    # domains = args.domains
    val_accuracies = []
    loss_values = []
    num_subject = args.target_subject

    for epoch in range(args.EPOCHS):
        model.train()
        running_loss = 0.0
        warm_up = 1 - math.exp(-epoch + warm) / (1 + math.exp(-epoch + warm))

        # warm_up = 0

        for data_batches in zip(*src_data_list, tgt_data):
            src_inputs = []
            src_labels = []
            for src_data in data_batches[:-1]:
                inputs, labels = src_data
                # inputs = EA(inputs) # Euclidean Alignment (Optional)
                src_inputs.append(inputs.to(args.device))
                src_labels.append(labels.to(args.device))

            inputs_tgt, labels_tgt = data_batches[-1]
            # inputs_tgt = EA(inputs_tgt) # Euclidean Alignment
            inputs_tgt, labels_tgt = inputs_tgt.to(args.device), labels_tgt.to(args.device)

            features = []
            outputs = []
            for src_input in src_inputs:
                feature, output = model(src_input, intermediate=True)
                features.append(feature - feature.mean(dim=0))
                outputs.append(output)

            feature_tgt, output_tgt = model(inputs_tgt, intermediate=True)
            feature_tgt = feature_tgt - feature_tgt.mean(dim=0)

            # Compute adaptive weights
            weights = compute_weights(features, feature_tgt)

            # Weighted cs_losses and cond_losses
            cs_losses = [weights[i] * CS(f, feature_tgt) for i, f in enumerate(features)]
            cond_losses = [weights[i] * CondCSD(f, feature_tgt, o, output_tgt) for i, (f, o) in enumerate(zip(features, outputs))]

            cs_losses_src = [CS(f1, f2) for i, f1 in enumerate(features) for f2 in features[i+1:]]
            cond_losses_src = [CondCSD(f1, f2, o1, o2) for i, (f1, o1) in enumerate(zip(features, outputs)) for f2, o2 in zip(features[i+1:], outputs[i+1:])]

            if len(cs_losses_src) > 0:
                marginal_ss = sum(cs_losses_src) / len(cs_losses_src)
            else:
                marginal_ss = 0  # if No. of source is less than 2
            marginal_st = sum(cs_losses)
            marginal_loss = marginal_st + marginal_ss

            if len(cond_losses_src) > 0:
                cond_ss = sum(cs_losses_src) / len(cs_losses_src)
            else:
                cond_ss = 0  # if source is less than 2
            cond_st = sum(cond_losses)
            cond_loss =  cond_st + cond_ss

            # Weighted cls_loss
            # cls_losses = [weights[i] * criterion(output, label) for i, (output, label) in
            #               enumerate(zip(outputs, src_labels))]
            # cls_loss = sum(cls_losses)

            cls_losses = [criterion(output, label) for i, (output, label) in
                            enumerate(zip(outputs, src_labels))]
            cls_loss = sum(cls_losses)/len(cls_losses)

            # cls_loss = criterion(outputs_combined, combined_labels)
            align_loss = alpha*(1-warm_up)*marginal_loss + beta*warm_up*cond_loss
            loss = cls_loss + align_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(src_data_list[0])
        loss_values.append(running_loss)
        print(f'Epoch {epoch + 1}/{args.EPOCHS}, Loss: {running_loss}')

        scheduler.step() # Update Scheduler

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

    # Calculate the average of the last 10 validation accuracies
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
    plt.title(f'Validation Accuracy for {args.target_subject},alpha={args.alpha},beta={args.beta},lr={args.lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss_values, label=f'Training Losses')
    plt.title(f'Losses for {args.model_type}, Sigma=Adapative')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.tight_layout()
    plt.show()

    return avg_val_acc

def auto_testing(data_trains, args):
    num_subjects = len(data_trains)
    avg_val_accuracies = []
    for src_idx in range(num_subjects):
        args['src_subject'] = src_idx
        for tgt_idx in range(num_subjects):
            args['tgt_subject'] = tgt_idx
            if src_idx != tgt_idx:
                print(f'Now testing with source subject {src_idx} and target subject {tgt_idx}')
                # avg_val_acc = single_train(data_trains[src_idx], data_trains[tgt_idx], args)
                # avg_val_accuracies.append((src_idx, tgt_idx, avg_val_acc))
    return avg_val_accuracies

def find_single_sources(data_trains, args):
    num_subjects = len(data_trains)
    best_sources = {}
    all_source_results = {}

    for tgt_idx in range(num_subjects):
        args['target_subject'] = tgt_idx
        tgt_data = data_trains[tgt_idx]
        source_performance = []

        # Evaluate each source individually
        for src_idx in range(num_subjects):
            if src_idx == tgt_idx:
                continue
            src_list = [data_trains[src_idx]]
            val_acc = multi_train(src_list, tgt_data, args)
            source_performance.append((src_idx, val_acc))

        # Sort sources by validation accuracy in descending order
        source_performance.sort(key=lambda x: x[1], reverse=True)

        # Select the top 4 sources
        top_sources = [src for src, _ in source_performance[:4]]
        best_sources[tgt_idx] = {
            'best_sources': top_sources,
            'best_val_acc': [val_acc for _, val_acc in source_performance[:4]]
        }

        # Store all source results
        all_source_results[tgt_idx] = source_performance

    return best_sources, all_source_results

if __name__ == "__main__":
    '''Argparse'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_num', type=int, default=9)
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--config_name', type=str, default='bcicompet2a_config') # change config name here
    aargs = parser.parse_args()

    # Config setting
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    config_name = aargs.config_name
    with open(os.path.join(config_dir, f'{config_name}.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        args = EasyDict(config)

    # Set Device
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = aargs.gpu_num
    args['device'] = get_device(aargs.gpu_num)
    cudnn.benchmark = False
    cudnn.fastest = False
    cudnn.deterministic = True

    # Update configs
    args.lr = float(args.lr)
    if args.downsampling != 0: args['sampling_rate'] = args.downsampling

    # Set SEED
    seed = args.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''Training'''
    alldata = {}
    for num_subject in range(args.num_subjects):
        args['target_subject'] = num_subject
        args.is_test = False
        print('--------------------------------------------------')
        print(config_name)
        print('--------------------------------------------------')

        alldata[num_subject] = get_dataset(config_name, args)

    data_trains = [
        DataLoader(alldata[i], batch_size=args.batch_size, shuffle=False)
        for i in range(args.num_subjects)
    ]

    '''BCIC 2a'''

    avg_val_accuracies0 = []

    args['target_subject'] = 0
    src_list = [data_trains[2], data_trains[3], data_trains[6], data_trains[7]]
    # weights = [0.27677, 0.26381, 0.273759, 0.1856567]
    val_acc = multi_train(src_list, data_trains[0],  args)
    avg_val_accuracies0.append(val_acc)

    args['target_subject'] = 1
    src_list = [data_trains[0], data_trains[3], data_trains[5], data_trains[7]]
    # weights = [0.18294, 0.27242, 0.331046, 0.21359]
    val_acc = multi_train(src_list, data_trains[1],  args)
    avg_val_accuracies0.append(val_acc)

    args['target_subject'] = 2
    src_list = [data_trains[1], data_trains[5], data_trains[7], data_trains[8]]
    # weights = [0.272128, 0.20550, 0.21584, 0.30651]
    val_acc = multi_train(src_list, data_trains[2],  args)
    avg_val_accuracies0.append(val_acc)

    args['target_subject'] = 3
    src_list = [data_trains[0], data_trains[2], data_trains[4], data_trains[7]]
    # weights = [0.298424, 0.190620, 0.31834, 0.19261]
    val_acc = multi_train(src_list, data_trains[3],  args)
    avg_val_accuracies0.append(val_acc)

    args['target_subject'] = 4
    src_list = [data_trains[2], data_trains[3], data_trains[5], data_trains[6]]
    # weights = [0.22861, 0.307805, 0.21555, 0.248024]
    val_acc = multi_train(src_list, data_trains[4], args)
    avg_val_accuracies0.append(val_acc)

    args['target_subject'] = 5
    src_list = [data_trains[2], data_trains[4], data_trains[6], data_trains[8]]
    # weights = [0.25731, 0.23099, 0.23798, 0.27371]
    val_acc = multi_train(src_list, data_trains[5],  args)
    avg_val_accuracies0.append(val_acc)

    args['target_subject'] = 6
    src_list = [data_trains[0], data_trains[4], data_trains[7], data_trains[8]]
    # weights = [0.263123, 0.217951, 0.25271, 0.26620]
    val_acc = multi_train(src_list, data_trains[6],  args)
    avg_val_accuracies0.append(val_acc)

    args['target_subject'] = 7
    src_list = [data_trains[2], data_trains[4], data_trains[5], data_trains[6]]
    # weights = [0.3231074, 0.2008151, 0.203564, 0.272512]
    val_acc = multi_train(src_list, data_trains[7],  args)
    avg_val_accuracies0.append(val_acc)

    args['target_subject'] = 8
    src_list = [data_trains[2], data_trains[4], data_trains[5], data_trains[6]]
    # weights = [0.206025, 0.1993181, 0.2720272, 0.3226287]
    val_acc = multi_train(src_list, data_trains[8],  args)
    avg_val_accuracies0.append(val_acc)

    print(f'2a validation accuracies: {avg_val_accuracies0}%')
    df = pd.DataFrame(avg_val_accuracies0, columns=['Accuracy'])
    df.to_csv('2a_selection_accuracies-w=200.csv', index=False)

    '''Cho 2017'''

    # Get data from all subject for Cho2017
    # near_subjects = [2, 28, 44, 30, 40, 42, 14, 4, 45, 17]
    # alldata = {}
    # for num_subject in near_subjects:
    #     args['target_subject'] = num_subject
    #     print('---------------------------------------------------')
    #     print(config_name)
    #     print(f'Loading subject S{num_subject:02d}')
    #     print('---------------------------------------------------')
    #     preprocessing_dict = args.preprocessing
    #     alldata[num_subject] = Cho2017(preprocessing_dict, num_subject, args.is_test)
    # #
    # # # Create DataLoader for near subjects
    # data_trains = [
    #     DataLoader(alldata[subject], batch_size=args.batch_size, shuffle=False)
    #     for subject in near_subjects
    # ]

    '''LOSO'''
    # Leave-one-subject-out training
    # avg_val_accuracies = []
    # model = get_teacher_model(args)
    # for tgt_subject in range(9):
    #     src_list = [data_trains[i] for i in range(9) if i != tgt_subject]
    #     src_indices = [i for i in range(9) if i != tgt_subject]
    #     print(f'Training with target subject {tgt_subject} and sources {src_indices}')
    #     args['target_subject'] = tgt_subject
    #     tgt_data = data_trains[tgt_subject]
    #     avg_val_acc = multi_train(src_list, tgt_data, args)
    #
    #     avg_val_accuracies.append((tgt_subject, tgt_subject, avg_val_acc))
    #
    # # Save accuracies to CSV
    # df = pd.DataFrame(avg_val_accuracies, columns=['Source', 'Target', 'Accuracy'])
    # df.to_csv('2a_baseline_accuracies.csv', index=False)

    '''Selection'''
    # avg_val_accuracies0 = []
    # start_time = time.time()
    # src_list = [data_trains[1], data_trains[2], data_trains[6], data_trains[7]]
    # args['target_subject'] = 0
    # val_acc = multi_train(src_list, data_trains[0], args)
    # avg_val_accuracies0.append(val_acc)
    # end_time = time.time()
    # print(f"Time taken for target subject 2: {end_time - start_time:.2f} seconds")
    #
    # start_time = time.time()
    # args['target_subject'] = 1
    # src_list = [data_trains[0], data_trains[2], data_trains[3], data_trains[6]]
    # val_acc = multi_train(src_list, data_trains[1], args)
    # avg_val_accuracies0.append(val_acc)
    # end_time = time.time()
    # print(f"Time taken for target subject 1: {end_time - start_time:.2f} seconds")
    #
    # start_time = time.time()
    # args['target_subject'] = 2
    # src_list = [data_trains[0], data_trains[1], data_trains[4], data_trains[5]]
    # val_acc = multi_train(src_list, data_trains[2], args)
    # avg_val_accuracies0.append(val_acc)
    # end_time = time.time()
    # print(f"Time taken for target subject 2: {end_time - start_time:.2f} seconds")
    #
    # start_time = time.time()
    # args['target_subject'] = 3
    # src_list = [data_trains[1], data_trains[2], data_trains[4], data_trains[9]]
    # val_acc = multi_train(src_list, data_trains[3], args)
    # avg_val_accuracies0.append(val_acc)
    # end_time = time.time()
    # print(f"Time taken for target subject 3: {end_time - start_time:.2f} seconds")
    #
    # start_time = time.time()
    # args['target_subject'] = 4
    # src_list = [data_trains[1], data_trains[2], data_trains[3], data_trains[8]]
    # val_acc = multi_train(src_list, data_trains[4], args)
    # avg_val_accuracies0.append(val_acc)
    # end_time = time.time()
    # print(f"Time taken for target subject 4: {end_time - start_time:.2f} seconds")
    #
    #
    # args['target_subject'] = 5
    # src_list = [data_trains[1], data_trains[2], data_trains[6], data_trains[9]]
    # val_acc = multi_train(src_list, data_trains[5], args)
    # avg_val_accuracies0.append(val_acc)
    #
    # args['target_subject'] = 6
    # src_list = [data_trains[0], data_trains[1], data_trains[5], data_trains[9]]
    # val_acc = multi_train(src_list, data_trains[6], args)
    # avg_val_accuracies0.append(val_acc)
    #
    # args['target_subject'] = 7
    # src_list = [data_trains[0], data_trains[3], data_trains[5], data_trains[8]]
    # val_acc = multi_train(src_list, data_trains[7], args)
    # avg_val_accuracies0.append(val_acc)
    #
    # args['target_subject'] = 8
    # src_list = [data_trains[0], data_trains[4], data_trains[7], data_trains[9]]
    # val_acc = multi_train(src_list, data_trains[8], args)
    # avg_val_accuracies0.append(val_acc)
    #
    # args['target_subject'] = 9
    # src_list = [data_trains[3], data_trains[5], data_trains[6], data_trains[8]]
    # val_acc = multi_train(src_list, data_trains[9], args)
    # avg_val_accuracies0.append(val_acc)
    #
    # print(f'Cho validation accuracies: {avg_val_accuracies0}%')
    # df = pd.DataFrame(avg_val_accuracies0, columns=['Accuracy'])
    # df.to_csv('Cho_selection_accuracies-b=0.csv', index=False)

    # Initialize a list to store results
    # import pandas as pd
    #
    # # Initialize a list to store results
    # results = []
    #
    # # Run the process 10 times
    # for run in range(10):
    #     print(f"Run {run + 1}/10")
    #
    #     # Initialize or reset variables/models if needed
    #     avg_val_accuracies = []
    #
    #     # Perform the EEG-UDA process (example: LOSO training)
    #     for tgt_subject in range(10):
    #         src_list = [data_trains[i] for i in range(10) if i != tgt_subject]
    #         src_indices = [i for i in range(10) if i != tgt_subject]
    #         print(f"Training with target subject {tgt_subject} and sources {src_indices}")
    #         args['target_subject'] = tgt_subject
    #         tgt_data = data_trains[tgt_subject]
    #         avg_val_acc = loso_train(src_list, tgt_data, args)
    #         avg_val_accuracies.append(avg_val_acc)
    #
    #     # Record the results for this run
    #     run_result = {"Run": run + 1}
    #     for subject, acc in enumerate(avg_val_accuracies):
    #         run_result[f"Subject_{subject}"] = acc
    #     run_result["Overall_Avg"] = sum(avg_val_accuracies) / len(avg_val_accuracies)
    #     results.append(run_result)
    #
    # # Save results to a CSV file
    # df = pd.DataFrame(results)
    # df.to_csv('EEG_UDA_results_with_subjects.csv', index=False)
    #
    # print("Results saved to 'EEG_UDA_results_with_subjects.csv'")