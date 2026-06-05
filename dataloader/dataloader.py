from glob import glob
from tqdm import tqdm
import mne
import torch
import scipy
import numpy as np
from dataloader.augmentation import cutcat,  cutcat_2
from typing import List, Union
import numpy as np
from mne.filter import resample
from torch.utils.data import Dataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import Preprocessor
from braindecode.datautil.preprocess import preprocess
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.windowers import create_windows_from_events
from filters import load_filterbank, butter_fir_filter
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataset import TensorDataset
from itertools import compress
from typing import Dict, Optional
from braindecode.preprocessing import (
    scale,
)
# from moabb.datasets import Cho2017

class BCICompet2aIV(torch.utils.data.Dataset):
    def __init__(self, args):
        
        '''
        * 769: Left
        * 770: Right
        * 771: foot
        * 772: tongue
        '''
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        
        self.data, self.label = self.get_brain_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)

        # transform to tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)

        return data, label

    def get_brain_data(self):
        filelist = sorted(glob(f'{self.base_path}/*T*.gdf')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E*.gdf'))
        
        label_filelist = sorted(glob(f'{self.base_path}/*T.mat')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E.mat'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            
            if idx != self.target_subject: continue
                    
            print(f'LOG >>> Filename: {filename}')
            
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, annot = mne.events_from_annotations(raw)
            
            raw.load_data()
            raw.filter(0.5, 40., fir_design='firwin')
            raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
            
            picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    eog=False,
                                    stim=False,
                                    exclude='bads')
            
            tmin, tmax = 0, 3
            if not self.is_test:
                event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10}) if idx != 3 \
                else dict({'769': 5,'770': 6,'771': 7,'772': 8})
            else:
                event_id = dict({'783': 7})
            
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin,
                                tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                preload=True)
            
            if self.downsampling != 0:
                epochs = epochs.resample(self.downsampling)
            self.fs = epochs.info['sfreq']
            
            epochs_data = epochs.get_data() * 1e6
            splited_data = []
            for epoch in epochs_data:
                normalized_data = exponential_moving_standardize(epoch, init_block_size=int(raw.info['sfreq'] * 3))
                # normalized_data = epoch
                splited_data.append(normalized_data)
            splited_data = np.stack(splited_data)
            splited_data = splited_data[:, np.newaxis, ...]
            
            label_list = scipy.io.loadmat(label_filelist[idx])['classlabel'].reshape(-1) - 1

            # Filter to include only labels 0 and 1
            mask = np.isin(label_list, [0, 1])
            splited_data = splited_data[mask]
            label_list = label_list[mask]

            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)
        # print(data.shape)
        return data, label

    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        # data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=8)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                             self.args.num_classes, ratio=8) # No augmentation implemented
        return data, label
    


class BaseDataModule:
    dataset = None
    train_dataset = None
    test_dataset = None

    def __init__(self, preprocessing_dict: Dict, subject_id: int):
        self.preprocessing_dict = preprocessing_dict
        self.subject_id = subject_id

    @staticmethod
    def _z_scale(X, X_test):
        for ch_idx in range(X.shape[1]):
            sc = StandardScaler()
            X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
            X_test[:, ch_idx, :] = sc.transform(X_test[:, ch_idx, :])
        return X, X_test

    @staticmethod
    def _z_scale_single(X):
        for ch_idx in range(X.shape[1]):
            sc = StandardScaler()
            X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
        return X

    @staticmethod
    def _make_tensor_dataset(X, y):
        return TensorDataset(torch.Tensor(X), torch.Tensor(y).type(torch.LongTensor))
    # def train_dataloader(self) -> DataLoader:
    #     return DataLoader(self.train_dataset,
    #                       batch_size=self.preprocessing_dict["batch_size"],
    #                       shuffle=True)
    #
    # def val_dataloader(self) -> DataLoader:
    #     return self.test_dataloader()
    #
    # def test_dataloader(self) -> DataLoader:
    #     return DataLoader(self.test_dataset,
    #                       batch_size=self.preprocessing_dict["batch_size"])



def load_cho2017(subject_ids: list, preprocessing_dict: Dict = None,
             verbose: str = "WARNING"):
    dataset = MOABBDataset(dataset_name="Cho2017", subject_ids=subject_ids)
    if preprocessing_dict.get("remove_artifacts", True):
        # find samples < 800 uV and save masks for later
        window_dataset = create_windows_from_events(dataset, preload=False)
        ds_masks = []
        for ds in window_dataset.datasets:
            clean_trial_mask = np.max(
                np.abs(ds.windows.load_data()._data), axis=(-2, -1)) < 800 * 1e-6
            ds_masks.append(clean_trial_mask)

    # channels = [
    #     "FC5", "FC1", "FC2", "FC6", "C3", "C4", "CP5", "CP1", "CP2", "CP6", "FC3",
    #     "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "FFC5h", "FFC3h",
    #     "FFC4h", "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h", "CCP3h", "CCP4h",
    #     "CCP6h", "CPP5h", "CPP3h", "CPP4h", "CPP6h", "FFC1h", "FFC2h", "FCC1h", "FCC2h",
    #     "CCP1h", "CCP2h", "CPP1h", "CPP2h",
    # ]

    preprocessors = [
        # Preprocessor("pick_channels", ch_names=channels, verbose=verbose),
        # Preprocessor(scale, factor=1e6, apply_on_array=True),  # from uV to V
        Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose)
    ]

    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        preprocessors.append(Preprocessor("filter", l_freq=l_freq, h_freq=h_freq,
                                          verbose=verbose))

    preprocess(dataset, preprocessors)

    # create windows
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples, preload=False
    )

    if preprocessing_dict.get("remove_artifacts", True):
        for (mask, ds) in zip(ds_masks, windows_dataset.datasets):
            ds.windows = ds.windows[mask]
            ds.y = list(compress(ds.y, mask))

    return windows_dataset

class Cho2017(BaseDataModule):
    all_subject_ids = list(range(1, 53))
    class_names = ["hand(L)", "hand(R)"]

    def __init__(self, preprocessing_dict, subject_id, is_test):
        super().__init__(preprocessing_dict, subject_id)
        self.is_test = is_test
        self.data, self.label = self.get_brain_data()

    def get_brain_data(self):
        self.dataset = load_cho2017(subject_ids=[self.subject_id], preprocessing_dict=self.preprocessing_dict)
        print(self.dataset)

        # Load the data
        X = self.dataset.datasets[0].windows.load_data()._data
        y = np.array(self.dataset.datasets[0].y)
        X = X[:, :64, :]
        # Scale data
        if self.preprocessing_dict["z_scale"]:
            X = BaseDataModule._z_scale_single(X)

        # Make dataset
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)

        data = X
        label = y

        B, C, W = data.shape
        data = data.reshape(B, 1, C, W)
        return data, label

    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        # data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=10)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                               2, ratio=10)
        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]

        if not self.is_test:
            data, label = self.augmentation(data, label)
        # transform to tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)
        return data, label




def get_dataset(config_name, args):
    
    if 'bcicompet2a_config' in config_name:
        dataset = BCICompet2aIV(args)
        if args['filter_bank']:
            #### FBCNet####
            # data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1], len(args['bank']),
            #                             dataset.data.shape[2], dataset.data.shape[3]))
            #
            # for num, Fband in enumerate(args['bank']):
            #     bw = np.array(Fband)
            #     filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
            #     X_filtered = np.zeros_like(dataset.data)
            #     for i, trial in enumerate(dataset.data):
            #         # filtering
            #         trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
            #         trail_filter = trail_filter.reshape(1, 22, 751)
            #         X_filtered[i, :, :, :] = trail_filter
            #     data_filterbank[:, :, num, :, :] = X_filtered

            #### IFNet####
            data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1],2*dataset.data.shape[2], dataset.data.shape[3]))

            for num, Fband in enumerate(args['bank']):
                bw = np.array(Fband)
                filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
                X_filtered = np.zeros_like(dataset.data)
                for i, trial in enumerate(dataset.data):
                    # filtering
                    trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
                    trail_filter = trail_filter.reshape(1, 22, 751)
                    X_filtered[i, :, :, :] = trail_filter
                data_filterbank[:, :, num*dataset.data.shape[2]: (num+1)*dataset.data.shape[2], :] = X_filtered
            dataset.data = data_filterbank
        else:
            dataset = dataset


    elif 'Cho_config' in config_name:
        preprocessing_dict = args.preprocessing
        subject_id = args.target_subject
        subject_id = subject_id + 1
        is_test = args.is_test
        dataset = Cho2017(preprocessing_dict, subject_id, is_test)

    else:
        raise Exception('get_dataset function Wrong dataset input!!!')

    return dataset
