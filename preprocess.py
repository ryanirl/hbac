#
# Will take ~25 minutes to process all of the data.
# 
from tqdm.auto import tqdm
import polars as pl
import numpy as np
import argparse
import torch
import h5py
import math
import os

from typing import Callable
from typing import Optional
from typing import List

from hbac.numerics import spectrogram
from hbac.numerics import butter_filter
from hbac.numerics import bin_array
from hbac.numerics import mad

from joblib import Parallel
from joblib import delayed
import threading

lock = threading.Lock()

SPEC_FREQ_CUTOFF = np.array([0.25, 40.0])
EEG_FREQ_CUTOFF = np.array([0.25, 50.0])
EKG_FREQ_CUTOFF = np.array([0.50, 20.0])

OUTPUT_PATHS = {
    "eeg": "eeg_down4x_f16.h5",
    "eeg_spectrogram": "eeg_spectrogram_hop44_nfft800_win256_f16.h5",
    "spectrogram": "kaggle_spectrogram_f16.h5"
}


@torch.no_grad()
def compute_spec(chain: np.ndarray) -> np.ndarray:
    t_chain = torch.Tensor(chain)
    t_chain = spectrogram(t_chain)
    t_chain = t_chain[:, :, 2:98]
    t_chain = torch.abs(t_chain) / 15
    t_chain = torch.log(t_chain.clip(math.exp(-4), math.exp(7)))
    t_chain = t_chain.mean(dim = 1)
    return t_chain.numpy()


def compute_spec_eeg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return butter_filter(a - b, cutoff_freq = SPEC_FREQ_CUTOFF, order = 5, btype = "bandpass")


def compute_eeg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eeg = butter_filter(a - b, cutoff_freq = np.array([0.25, 50]), btype = "bandpass")
    eeg = bin_array(eeg, bin_size = 4, mode = "reflect").mean(axis = -1)
    return eeg


def prepare_banana_montage(df_eeg: pl.DataFrame, fn: Callable) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy() 
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3  = df_eeg["F3"].to_numpy()
    F4  = df_eeg["F4"].to_numpy()
    F7  = df_eeg["F7"].to_numpy()
    F8  = df_eeg["F8"].to_numpy()
    C3  = df_eeg["C3"].to_numpy()
    C4  = df_eeg["C4"].to_numpy()
    P3  = df_eeg["P3"].to_numpy()
    P4  = df_eeg["P4"].to_numpy()
    T3  = df_eeg["T3"].to_numpy()
    T4  = df_eeg["T4"].to_numpy()
    T5  = df_eeg["T5"].to_numpy()
    T6  = df_eeg["T6"].to_numpy()
    O1  = df_eeg["O1"].to_numpy()
    O2  = df_eeg["O2"].to_numpy()
    
    ll = np.stack([(fn(Fp1, F7), fn(F7, T3), fn(T3, T5), fn(T5, O1))])
    lp = np.stack([(fn(Fp1, F3), fn(F3, C3), fn(C3, P3), fn(P3, O1))])
    rp = np.stack([(fn(Fp2, F4), fn(F4, C4), fn(C4, P4), fn(P4, O2))])
    rl = np.stack([(fn(Fp2, F8), fn(F8, T4), fn(T4, T6), fn(T6, O2))])
    chain = np.stack([ll, lp, rp, rl])[:, 0]

    return chain


def compute_eeg_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    chain = prepare_banana_montage(df_eeg, fn = compute_spec_eeg)
    mads = mad(chain, axis = -1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)
    chain = compute_spec(chain)

    
    return chain


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fz  = df_eeg["Fz"].to_numpy()
    Cz  = df_eeg["Cz"].to_numpy()
    Pz  = df_eeg["Pz"].to_numpy()
    ekg = df_eeg["EKG"].to_numpy()

    ekg = butter_filter(ekg, cutoff_freq = np.array([0.50, 20.0]), btype = "bandpass")
    ekg = bin_array(ekg, bin_size = 4, mode = "reflect").mean(axis = -1)
    ekg = ekg.reshape(1, -1)

    mid = np.stack([compute_eeg(Fz, Cz), compute_eeg(Cz, Pz)])
    
    chain = prepare_banana_montage(df_eeg, fn = compute_eeg)

    # Stored in reverse order. To access again would be:
    # ekg = out[0].reshape(1, -1)
    # mid = out[1:3].reshape(2, -1)
    # chain = out[3:].reshape(16, -1).reshape(4, 4, -1)
    out = np.concatenate([ 
        ekg.reshape(1, -1),
        mid.reshape(2, -1), 
        chain.reshape(16, -1)
    ], axis = 0)

    return out


def process_kaggle_spec(spec: np.ndarray) -> np.ndarray:
    spec[np.isnan(spec) | np.isinf(spec)] = 0
    spec = spec[:, 1:]
    spec = np.stack([
        spec[:,   0:100].T,
        spec[:, 100:200].T,
        spec[:, 200:300].T,
        spec[:, 300:400].T,
    ])

    # I store everything as f16 to save on space and loading time.
    spec = spec.astype(np.float16)

    return spec


def compute_kaggle_spec_from_file(filepath: str) -> np.ndarray:
    pl_spec = pl.read_parquet(filepath).fill_null(0)
    spec = process_kaggle_spec(pl_spec.to_numpy().astype(np.float32))
    return spec
    

def compute_eeg_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_spec_chain(df_eeg)
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    modalities = ["eeg", "eeg_spectrogram", "spectrogram"]

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i", "--data-dir", type = str, required = True, metavar = "",
        help = "Path to the directory containing the Kaggle HMS-HBAC dataset."
    )
    required.add_argument(
        "-m", "--modality", type = str, required = True, metavar = "", choices = modalities,
        help = f"The modality of the data to preprocess. Choices are ({', '.join(modalities)})"
    )
    required.add_argument(
        "--num-workers", type = int, required = False, metavar = "", default = 4,
        help = f"The number of workers to use for parallel processing."
    )
    args = parser.parse_args(argv)

    assert args.modality in modalities # JIC

    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_file = os.path.join(args.data_dir, OUTPUT_PATHS[args.modality])

    eeg_dir = os.path.join(args.data_dir, "train_eegs")
    spec_dir = os.path.join(args.data_dir, "train_spectrograms")

    # Validate the input. Should probably not use assertions. 
    assert os.path.exists(eeg_dir)
    assert os.path.exists(spec_dir)
    assert not os.path.exists(output_file)

    df_train = pl.read_csv(os.path.join(args.data_dir, "train.csv"))

    # Where to write the data.
    db = h5py.File(output_file, "w")

    # Helper function defined within main to give relative-global access to `out`.
    def worker(modality_id: int, modality: str) -> None:
        if modality == "eeg":
            chain = compute_eeg_from_file(os.path.join(eeg_dir, f"{modality_id}.parquet"))
        elif modality == "eeg_spectrogram":
            chain = compute_eeg_spec_from_file(os.path.join(eeg_dir, f"{modality_id}.parquet"))
        else: # Kaggle spectrogram
            chain = compute_kaggle_spec_from_file(os.path.join(spec_dir, f"{modality_id}.parquet"))

        with lock:
            db.create_dataset(str(modality_id), data = chain, dtype = np.float16)

    if args.modality in ["eeg", "eeg_spectrogram"]:
        ids = df_train["eeg_id"].unique()
    else: # Generate kaggle spectrogram
        ids = df_train["spectrogram_id"].unique()

    with Parallel(n_jobs = args.num_workers, require = "sharedmem", backend = "threading") as parallel:
        parallel(delayed(worker)(ids[i], args.modality) for i in tqdm(range(len(ids))))
    
    db.close()


if __name__ == "__main__":
    main()


