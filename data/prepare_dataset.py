import argparse
import os
import h5py
from tqdm import tqdm

from utils import load
from pathlib import Path
from random import shuffle

def prepare_hdf_file(hdf_dir, pair_list,
                     clean_root, noisy_root,
                     sr, channels):

    # Create HDF file
    with h5py.File(hdf_dir, "w") as f:
        f.attrs["sr"] = sr
        f.attrs["channels"] = channels

        for clean, noisy in tqdm(pair_list):
            idx = Path(clean).name
            # Load mix
            clean_audio, _ = load(clean_root / clean, sr=sr,
                                  mono=(channels == 1))
            noisy_audio, _ = load(noisy_root / noisy, sr=sr,
                                  mono=(channels == 1))
            # audio off 1 sec prefix and suffix from noisy
            DEFAULT_NOISY_PAD_SAMPLES = int(sr * 1.0)  # seconds
            pad_samples = DEFAULT_NOISY_PAD_SAMPLES
            noisy_audio = noisy_audio[:, pad_samples:-pad_samples]
            # TODO normalize?
            if noisy_audio.shape[1] != clean_audio.shape[1]:
                print(f"Warning: noise and clean audio have different length: {idx}")
                continue

            # Add to HDF5 file
            grp = f.create_group(idx)
            grp.create_dataset("targets", shape=clean_audio.shape,
                               dtype=clean_audio.dtype, data=clean_audio)
            grp.create_dataset("inputs", shape=noisy_audio.shape,
                               dtype=noisy_audio.dtype, data=noisy_audio)
            # lengths are identical
            grp.attrs["length"] = clean_audio.shape[1]


def prepare_splits(hdf_root:Path, clean_file_list, noisy_file_list, sr, channels):
    # Create folder if it did not exist before
    if not os.path.exists(hdf_root):
        os.makedirs(hdf_root)

    clean_file_list = Path(clean_file_list)
    noisy_file_list = Path(noisy_file_list)

    clean_root = clean_file_list.parent
    noisy_root = noisy_file_list.parent

    print("Adding audio files to dataset (preprocessing)...")
    with open(clean_file_list, "r") as cf:
        clean_data = (l.strip() for l in cf.readlines())
    with open(noisy_file_list, "r") as nf:
        noisy_data = (l.strip() for l in nf.readlines())

    full_file_list = list(zip(clean_data, noisy_data))
    shuffle(full_file_list)
    l = len(full_file_list)
    s1 = int(l*0.70)
    s2 = s1 + int((l-s1)*2.0/3)

    train_list = full_file_list[:s1]
    val_list = full_file_list[s1:s2]
    test_list = full_file_list[s2:]

    prepare_hdf_file(hdf_root / 'train.hdf5', train_list, clean_root, noisy_root, sr, channels)
    prepare_hdf_file(hdf_root / 'val.hdf5', val_list, clean_root, noisy_root, sr, channels)
    prepare_hdf_file(hdf_root / 'test.hdf5', test_list, clean_root, noisy_root,  sr, channels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default="/home/kiefer/media/Transport/tetra",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="hdf",
                        help='Dataset path')
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=1,
                        help="Number of input audio channels")
    parser.add_argument('--clean_list', type=str, default="voice120hours.list",
                        help='List of clean files')
    parser.add_argument('--noise_list', type=str, default="noise120hours.list",
                        help='List of noise files')

    args = parser.parse_args()

    data_dir = Path(args.dataset_dir)
    prepare_splits(Path(args.hdf_dir),
                   data_dir / args.clean_list,
                   data_dir / args.noise_list, 44100, 1)