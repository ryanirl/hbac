from collections import defaultdict
from tqdm.auto import tqdm
import polars as pl
import numpy as np
import dataclasses
import random
import math
import cv2
import os

from sklearn.model_selection import GroupKFold

from hbac.numerics import mad
from hbac.datasets.utils import LazyH5Database
from hbac.datasets.utils import EegDatabase

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from typing import Optional
from typing import Tuple

LABELS = [
    "seizure_vote", 
    "lpd_vote", 
    "gpd_vote", 
    "lrda_vote", 
    "grda_vote", 
    "other_vote"
]

CHAINS = {
    "LL": ["Fp1", "F7", "T3", "T5", "O1"],
    "LP": ["Fp1", "F3", "C3", "P3", "O1"],
    "RP": ["Fp2", "F4", "C4", "P4", "O2"],
    "RR": ["Fp2", "F8", "T4", "T6", "O2"]
}

DB_EEG = EegDatabase("./data/eeg_down4x_f16.h5")
DB_EEG_SPEC = LazyH5Database("./data/eeg_spectrogram_hop44_nfft800_win256_f16.h5") 
DB_SPEC = LazyH5Database("./data/kaggle_spectrogram_f16.h5")

EEG_DOWNSAMPLE = 4
EEG_LENGTH = 10_000 // EEG_DOWNSAMPLE
EEG_FS = 200 // EEG_DOWNSAMPLE

SPEC_HEIGHT = 96
SPEC_WIDTH = 224

SPEC_FREQ_CUTOFF = np.array([0.25, 40.0])
EEG_FREQ_CUTOFF = np.array([0.25, 50.0])
EKG_FREQ_CUTOFF = np.array([0.50, 20.0])


def process_eeg(eeg: np.ndarray, mid: np.ndarray, ekg: np.ndarray, offset: Optional[int] = None, *, copy: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    if copy:
        eeg = eeg.copy()
        mid = mid.copy()
        ekg = ekg.copy()

    eeg = eeg.astype(np.float32)
    mid = mid.astype(np.float32)
    ekg = ekg.astype(np.float32)

    if offset is not None:
        offset_l = int(offset * EEG_FS)
        offset_r = offset_l + EEG_LENGTH 
        eeg = eeg[:, :, offset_l : offset_r] 
        ekg = ekg[:, offset_l : offset_r] 
        mid = mid[:, offset_l : offset_r] 

    # Ensure all of the values are real.
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    ekg[np.isnan(ekg) | np.isinf(ekg)] = 0
    mid[np.isnan(mid) | np.isinf(mid)] = 0

    eeg = eeg - eeg.mean(axis = -1, keepdims = True)
    mid = mid - mid.mean(axis = -1, keepdims = True)

    # Robust estimate of standard deviation.
    mad_std = mad(eeg, axis = -1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = eeg / mad_std
    mid = mid / mad_std
    eeg = eeg.clip(-10, 10)
    mid = mid.clip(-10, 10)

    ekg = ekg / (mad(ekg, axis = -1).reshape(-1) + 1e-5)

    return eeg, mid, ekg


def process_eeg_spec(spec: np.ndarray, offset: Optional[int] = None, *, copy: bool = False) -> np.ndarray:
    """
    """
    if copy:
        spec = spec.copy()

    spec = spec.astype(np.float32)

    if offset is not None:
        offset_l = math.floor((offset * 200) / 44) # Hop length
        offset_r = offset_l + 228
        spec = spec[:, :, offset_l : offset_r].copy()

    spec = spec[:, :, 2:-2] # 100 -> 96 height.

    spec[np.isnan(spec) | np.isinf(spec)] = 0
    spec = spec + 1
    spec = spec.reshape(4, 96, 224)

    return spec


def process_spec(spec: np.ndarray, offset: Optional[int] = None, *, copy: bool = False) -> np.ndarray:
    """
    """
    if copy:
        spec = spec.copy()

    spec = spec.astype(np.float32)

    if offset is not None:
        offset_l = int(offset // 2) 
        offset_r = offset_l + 300

        spec = spec[:, :, offset_l : offset_r]

    spec[np.isnan(spec) | np.isinf(spec)] = 0
    spec = spec[:, :, 2:-2] # 228 -> 224 width
    spec = np.log(spec.clip(np.exp(-4), np.exp(7)))

    spec = spec - spec.mean(axis = (1, 2), keepdims = True)
    spec = spec / (spec.std(axis = (1, 2), keepdims = True) + 1e-5)

    spec = spec.transpose(1, 2, 0)
    spec = cv2.resize(spec, (SPEC_WIDTH, SPEC_HEIGHT), interpolation = cv2.INTER_CUBIC)
    spec = spec.transpose(2, 0, 1)

    return spec


@dataclasses.dataclass
class HmsDataModule:
    # HmsDataset parameters
    data_dir: str = "./data/"
    data_type: str = "eeg"
    fold: int = 0
    n_folds: int = 5
    count_type: str = "all"

    # Dataloader parameters
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = True

    def get_train_loader(self):
        train_dataset = HmsDataset(
            data_dir = self.data_dir,
            data_type = self.data_type,
            split = "train",
            fold = self.fold,
            n_folds = self.n_folds,
            count_type = self.count_type
        )
        train_dataloader = DataLoader(
            dataset = train_dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            drop_last = self.drop_last
        )
        return train_dataloader

    def get_validation_loader(self):
        val_dataset= HmsDataset(
            data_dir = self.data_dir,
            data_type = self.data_type,
            split = "test",
            fold = self.fold,
            n_folds = self.n_folds,
            count_type = "all"
        )
        val_dataloader = DataLoader(
            dataset = val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            drop_last = False,
        )
        return val_dataloader

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return self.get_train_loader(), self.get_validation_loader()


def get_fold(
    ids: np.ndarray,
    patient_ids: np.ndarray,
    split: str = "train", 
    n_folds: int = 5, 
    fold: int = 0
) -> np.ndarray:
    """
    """
    kfold = GroupKFold(n_splits = n_folds)
    train_inds, test_inds = list(kfold.split(ids, groups = patient_ids))[fold]

    if split == "train":
        inds = train_inds
    elif split == "test":
        inds = test_inds
    else:
        raise ValueError("'split' must be either 'train' or 'test'.")

    return ids[inds]


class HmsDataset(Dataset):
    splits = ["train", "test"]
    data_types = ["eeg", "eeg_spectrogram", "spectrogram", "multimodal"]
    count_types = ["all", "upper", "lower"]

    def __init__(
        self, 
        data_dir: str, 
        data_type: str = "eeg",
        split: str = "train", 
        fold: int = 0,
        n_folds: int = 5,
        count_type: str = "all",
        transforms: bool = True
    ) -> None:
        """ """
        if split not in self.splits:
            raise ValueError(f"'split' must be one of [{', '.join(self.splits)}].")

        if data_type not in self.data_types:
            raise ValueError(f"'data_type' must be one of [{', '.join(self.data_types)}].")

        if count_type not in self.count_types:
            raise ValueError(f"'count_type' must be one of [{', '.join(self.count_types)}].")

        self.data_dir = data_dir
        self.data_type = data_type
        self.split = split
        self.fold = fold
        self.n_folds = n_folds
        self.count_type = count_type
        self.transforms = transforms

        # Metadata. 
        df_train = pl.read_csv(os.path.join(data_dir, "train.csv"))

        labels = df_train[LABELS].to_numpy()
        counts = labels.sum(axis = 1)
        labels = labels / counts[:, None]

        self.eeg_labels_all = defaultdict(list)
        self.eeg_labels_upper = defaultdict(list)
        self.eeg_labels_lower = defaultdict(list)
        for i in tqdm(range(len(df_train))):
            df_row = df_train[i]
            eeg_id = df_row["eeg_id"].item()
            sample = (
                labels[i], 
                eeg_id,
                df_row["spectrogram_id"].item(),
                df_row["eeg_label_offset_seconds"].item(),
                df_row["spectrogram_label_offset_seconds"].item(),
                df_row["patient_id"].item(),
                counts[i]
            )

            # Very useful for training methods using the bimodal labels. 
            if counts[i] > 9:
                self.eeg_labels_upper[eeg_id].append(sample)
            else:
                self.eeg_labels_lower[eeg_id].append(sample)

            self.eeg_labels_all[eeg_id].append(sample)

        # Always CV split on the same data for consistancy.
        _ids = np.array(list(self.eeg_labels_all.keys()))
        _pids = np.array([self.eeg_labels_all[ID][0][5] for ID in _ids])

        # GroupKFold on `eeg_id` (ids), grouped by `patient_id` (pid).
        self.ids_all = get_fold(_ids, _pids, split, n_folds, fold)
        self.ids_upper = np.intersect1d(self.ids_all, np.array(list(self.eeg_labels_upper.keys())))
        self.ids_lower = np.intersect1d(self.ids_all, np.array(list(self.eeg_labels_lower.keys())))

        # Now determine what subset of data will be used. See the documentation
        # above for more information on what this means. 
        if self.count_type == "all":
            self.ids = self.ids_all
            self.data = self.eeg_labels_all
        elif self.count_type == "upper":
            self.ids = self.ids_upper
            self.data = self.eeg_labels_upper
        else: # Lower
            self.ids = self.ids_lower
            self.data = self.eeg_labels_lower

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        eeg_id = self.ids[idx]

        # During training, randomly select an eeg and spectrogram offset from
        # the current eeg id. 
        y, eeg_id, spc_id, eeg_offset, spc_offset, pid, count = random.choice(self.data[eeg_id])

        if self.data_type == "spectrogram":
            x = self.get_spectrogram(spc_id, spc_offset)
            return x, y, count, eeg_id

        elif self.data_type == "eeg_spectrogram":
            x = self.get_eeg_spectrogram(eeg_id, eeg_offset)
            return x, y, count, eeg_id 

        elif self.data_type == "eeg":
            x = self.get_eeg(eeg_id, eeg_offset)
            return x, y, count, eeg_id 

        elif self.data_type == "multimodal":
            eeg = self.get_eeg(eeg_id, eeg_offset)
            eeg_spec = self.get_eeg_spectrogram(eeg_id, eeg_offset)
            spec = self.get_spectrogram(spc_id, spc_offset)
            return eeg, y, count, eeg_id, eeg_spec, spec 

        else:
            raise ValueError(
                "'data_type' must be either 'spectrogram', 'eeg_spectrogram', or 'eeg'."
            )

    def get_spectrogram(self, spc_id: int, spc_offset: int) -> torch.Tensor:
        """
        """
        spec = DB_SPEC[str(spc_id)]
        spec = process_spec(spec, spc_offset, copy = True)

        if (self.split == "train") and (self.transforms):
            if random.uniform(0, 1) < 0.5:
                spec = spec[::-1].copy()

        return spec

    def get_eeg(self, eeg_id: int, eeg_offset: int) -> torch.Tensor:
        """
        """
        eeg, mid, ekg = DB_EEG[str(eeg_id)]
        eeg, mid, ekg = process_eeg(eeg, mid, ekg, eeg_offset, copy = True)

        # Transforms
        if (self.split == "train") and (self.transforms):
            eeg, mid = eeg_augmentations(eeg, mid)

        eeg = eeg.reshape(16, -1)
        eeg = np.concatenate([eeg, mid, ekg], axis = 0)
        eeg = eeg.reshape(19, EEG_LENGTH)

        return eeg

    def get_eeg_spectrogram(self, eeg_id: int, eeg_offset: int) -> torch.Tensor:
        """
        """
        spec = DB_EEG_SPEC[str(eeg_id)]
        spec = process_eeg_spec(spec, eeg_offset, copy = True)

        if (self.split == "train") and (self.transforms):
            if random.uniform(0, 1) < 0.5:
                spec = spec[::-1].copy()

        return spec


def eeg_augmentations(eeg: np.ndarray, mid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eeg = eeg.reshape(16, -1)

    # Randomly zero up to four channels
    if random.uniform(0, 1) < 0.50:
        if random.uniform(0, 1) < (2 / 18):
            inds = np.random.randint(0, 1, size = (1))
            mid[inds] = 0
        else:
            size = np.random.randint(0, 4, size = (1))
            inds = np.random.randint(0, 16, size = size)
            eeg[inds] = 0

    eeg = eeg.reshape(4, 4, -1)

    if random.uniform(0, 1) < 0.5:
        eeg = eeg[::-1]

    if random.uniform(0, 1) < 0.5:
        eeg = -eeg[:, ::-1]
        mid = -mid[::-1]

    return eeg.copy(), mid.copy()


if __name__ == "__main__":
    dataset = HmsDataset(
        data_dir = "./data/",
        data_type = "eeg_spectrogram",
        split = "train",
        n_folds = 5,
        fold = 0,
    )

    print(dataset[0][0].shape)


