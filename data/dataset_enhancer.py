from pathlib import Path

import h5py
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset

from data.prepare_dataset import prepare_hdf_file

class EnhancementDataset(Dataset):
    def __init__(self, hdf_dir,
                 partition, sr, channels,
                 shapes, random_hops,
                 audio_transform=None, in_memory=False):
        '''
        Initialises a source separation dataset
        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(EnhancementDataset, self).__init__()

        self.hdf_dataset = None
        hdf_dir = Path(hdf_dir)
        hdf_dir.mkdir(parents=True, exist_ok=True)
        self.hdf_dir = Path(hdf_dir) / partition + ".hdf5"

        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory

        # Check if HDF file exists
        if not self.hdf_dir.exists():
            raise ValueError("HDF file does not exist")

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_dir, "r") as f:
            if f.attrs["sr"] != sr or \
                    f.attrs["channels"] != channels:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate and channel are not as expected. Did you load an out-dated HDF file?")

        # HDF FILE READY

        # SET SAMPLING POSITIONS

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]


    def __getitem__(self, index):
        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver)

        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1]

        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]

        # Determine position where to start targets
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(audio_length - self.shapes["output_frames"] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos = index * self.shapes["output_frames"]

        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = self.hdf_dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            targets = np.pad(targets, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)

        return audio, targets

    def __len__(self):
        return self.length
