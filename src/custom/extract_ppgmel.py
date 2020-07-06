"""Modified from https://github.com/NVIDIA/tacotron2"""
# modify by common/data_utils.py

import pickle
import random
import numpy as np
import torch
import torch.utils.data

import sys
sys.path.insert(0, '/home/fuu/sdb/fac-via-ppg/src')

from common.utils import load_filepaths
from common.utterance import Utterance
from common import layers
from ppg import DependenciesPPG
from scipy.io import wavfile
from common import feat
from common import ppg
from common.hparams_628 import create_hparams




# First order, dx(t) = 0.5(x(t + 1) - x(t - 1))
DELTA_WIN = [0, -0.5, 0.0, 0.5, 0]
# Second order, ddx(t) = 0.5(dx(t + 1) - dx(t - 1)) = 0.25(x(t + 2) - 2x(t)
# + x(t - 2))
ACC_WIN = [0.25, 0, -0.5, 0, 0.25]


def get_ppg(wav_path, deps):
    fs, wav = wavfile.read(wav_path)
    wave_data = feat.read_wav_kaldi_internal(wav, fs)
    seq = ppg.compute_full_ppg_wrapper(wave_data, deps.nnet, deps.lda, 10)
    return seq


def compute_dynamic_vector(vector, dynamic_win, frame_number):
    """Modified from https://github.com/CSTR-Edinburgh/merlin/blob/master
    /srcfrontend/acoustic_base.py
    Compute dynamic features for a data vector.
    Args:
        vector: A T-dim vector.
        dynamic_win: What type of dynamic features to compute. See DELTA_WIN
        and ACC_WIN.
        frame_number: The dimension of 'vector'.
    Returns:
        Dynamic feature vector.
    """
    vector = np.reshape(vector, (frame_number, 1))

    win_length = len(dynamic_win)
    win_width = int(win_length / 2)
    temp_vector = np.zeros((frame_number + 2 * win_width, 1))
    dynamic_vector = np.zeros((frame_number, 1))

    temp_vector[win_width:frame_number + win_width] = vector
    for w in range(win_width):
        temp_vector[w, 0] = vector[0, 0]
        temp_vector[frame_number + win_width + w, 0] = vector[
            frame_number - 1, 0]

    for i in range(frame_number):
        for w in range(win_length):
            dynamic_vector[i] += temp_vector[i + w, 0] * dynamic_win[w]

    return dynamic_vector


def compute_dynamic_matrix(data_matrix, dynamic_win):
    """Modified from https://github.com/CSTR-Edinburgh/merlin/blob/master
    /srcfrontend/acoustic_base.py
    Compute dynamic features for a data matrix. Calls compute_dynamic_vector
    for each feature dimension.
    Args:
        data_matrix: A (T, D) matrix.
        dynamic_win: What type of dynamic features to compute. See DELTA_WIN
        and ACC_WIN.
    Returns:
        Dynamic feature matrix.
    """
    frame_number, dimension = data_matrix.shape
    dynamic_matrix = np.zeros((frame_number, dimension))

    # Compute dynamic feature dimension by dimension
    for dim in range(dimension):
        dynamic_matrix[:, dim:dim + 1] = compute_dynamic_vector(
            data_matrix[:, dim], dynamic_win, frame_number)

    return dynamic_matrix


def compute_delta_acc_feat(matrix, is_delta=False, is_acc=False):
    """A wrapper to compute both the delta and delta-delta features and
    append them to the original features.
    Args:
        matrix: T*D matrix.
        is_delta: If set to True, compute delta features.
        is_acc: If set to True, compute delta-delta features.
    Returns:
        matrix: T*D (no dynamic feature) | T*2D (one dynamic feature) | T*3D
        (two dynamic features) matrix. Original feature matrix concatenated
    """
    if not is_delta and is_acc:
        raise ValueError('To use delta-delta feats you have to also use '
                         'delta feats.')
    if is_delta:
        delta_mat = compute_dynamic_matrix(matrix, DELTA_WIN)
    if is_acc:
        acc_mat = compute_dynamic_matrix(matrix, ACC_WIN)
    if is_delta:
        matrix = np.concatenate((matrix, delta_mat), axis=1)
    if is_acc:
        matrix = np.concatenate((matrix, acc_mat), axis=1)
    return matrix


def append_ppg(feats, f0):
    """Append log F0 and its delta and acc

    Args:
        feats:
        f0:

    Returns:

    """
    num_feats_frames = feats.shape[0]
    num_f0_frames = f0.shape[0]
    final_num_frames = min([num_feats_frames, num_f0_frames])
    feats = feats[:final_num_frames, :]
    f0 = f0[:final_num_frames]
    lf0 = np.log(f0 + np.finfo(float).eps)  # Log F0.
    lf0 = lf0.reshape(lf0.shape[0], 1)  # Convert to 2-dim matrix.
    lf0 = compute_delta_acc_feat(lf0, True, True)
    return np.concatenate((feats, lf0), axis=1)


class PPGMelLoader_x(torch.utils.data.Dataset):
    """Loads [ppg, mel] pairs."""

    def __init__(self, data_utterance_paths, cache_path, hparams, bs, loop):
        """Data loader for the PPG->Mel task.

        Args:
            data_utterance_paths: A text file containing a list of file paths.
            hparams: The hyper-parameters.
        """
        self.data_utterance_paths = load_filepaths(data_utterance_paths)
        self.max_wav_value = 32768.0
        self.sampling_rate = hparams.sampling_rate
        self.is_full_ppg = True
        self.is_append_f0 = False
        self.is_cache_feats = True
        self.feats_cache_path = cache_path
        self.ppg_subsampling_factor = 1
        self.ppg_deps = DependenciesPPG()
        # 20 data = n(4) * b(5)
        self.n = int(bs) - 1
        self.b = 5
        self.l = int(loop) - 1

        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_acoustic_feat_dims, self.sampling_rate,
            hparams.mel_fmin, hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.data_utterance_paths)
        
        if self.n > 0:
            with open(self.feats_cache_path, 'rb') as f:
                data = pickle.load(f)
            self.ppg_sequences = data[0]
            self.acoustic_sequences = data[1]
        else:
            self.ppg_sequences = []
            self.acoustic_sequences = []

        for utterance_path in self.data_utterance_paths[self.n * self.b + self.l * 20 : (self.n+1) * self.b + self.l * 20]:

            ppg_feat_pair = self.extract_utterance_feats(utterance_path,
                                                self.is_full_ppg)
            self.ppg_sequences.append(ppg_feat_pair[0].astype(
                np.float32))
            self.acoustic_sequences.append(ppg_feat_pair[1])
        
        if self.is_cache_feats:
            print('Caching data to %s.' % self.feats_cache_path)
            with open(self.feats_cache_path, 'wb+') as f:
                pickle.dump([self.ppg_sequences, self.acoustic_sequences], f) 



    def extract_utterance_feats(self, data_utterance_path, is_full_ppg=False):
        """Get PPG and Mel (+ optional F0) for an utterance.

        Args:
            data_utterance_path: The path to the data utterance protocol buffer.
            is_full_ppg: If True, will use the full PPGs.

        Returns:
            feat_pairs: A list, each is a [pps, mel] pair.
        """
        utt = Utterance()
        ppg_deps = DependenciesPPG()
        fs, wav = wavfile.read(data_utterance_path)
        utt.fs = fs
        utt.wav = wav
        utt.ppg = get_ppg(data_utterance_path, ppg_deps)

        audio = torch.FloatTensor(utt.wav.astype(np.float32))
        fs = utt.fs

        if fs != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                fs, 22050))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        # (1, n_mel_channels, T)
        acoustic_feats = self.stft.mel_spectrogram(audio_norm)
        # (n_mel_channels, T)
        acoustic_feats = torch.squeeze(acoustic_feats, 0)
        # (T, n_mel_channels)
        acoustic_feats = acoustic_feats.transpose(0, 1)
        return [utt.ppg, acoustic_feats]

    def __getitem__(self, index):
        """Get a new data sample in torch.float32 format.

        Args:
            index: An int.

        Returns:
            T*D1 PPG sequence, T*D2 mels
        """
        if self.ppg_subsampling_factor == 1:
            curr_ppg = self.ppg_sequences[index]
        else:
            curr_ppg = self.ppg_sequences[index][
                       0::self.ppg_subsampling_factor, :]

        return torch.from_numpy(curr_ppg), self.acoustic_sequences[index]

    def __len__(self):
        return len(self.ppg_sequences)


if __name__ == '__main__':

    training_files_path = "/home/fuu/sdb/fac-via-ppg/data/filelists/cmu/bdl_train_list.txt"
    training_files_path = "/home/fuu/sdb/fac-via-ppg/data/filelists/cmu/bdl_val_list.txt"
    txtname = sys.argv[2]

    # cache_path = "/home/fuu/sdb/data/cmu/ppg_mel_pair/bdl_train/file_20_" + txtname + ".txt"            # file: 45
    cache_path = "/home/fuu/sdb/data/cmu/ppg_mel_pair/bdl_val/file_20_" + txtname + ".txt"                # file: 11
    hparams = create_hparams()
    PPGMelLoader_x(training_files_path, cache_path, hparams, sys.argv[1], sys.argv[2])

    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    print("#### shape of data:", np.shape(data), " ####")
