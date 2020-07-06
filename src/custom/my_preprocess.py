# Library
import sys
sys.path.insert(0, "/home/fuu/sdb/fac-via-ppg/src")

import os
import wave
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import common.data_utils as du
import ppg
import pickle

from common.hparams_628 import create_hparams_stage
from script.train_ppg2mel_628 import load_model
from ppg import DependenciesPPG
import seaborn as sns
from scipy.io import wavfile
from common import feat
from common.layers import TacotronSTFT
from waveglow.denoiser import Denoiser

from kaldi.matrix.sparse import SparseMatrix
from kaldi.matrix import Matrix, Vector
from kaldi.matrix.common import MatrixTransposeType

"""
生成音频文件列表txt的相关函数
read_list()
segment_dataset()
save_listtxt()
"""
# 读取文件路径到列表
def read_listdir(path):
    file_name_list = os.listdir(path)
    file_path_list = []
    for fn in file_name_list:
        fp = os.path.join(path, fn)
        file_path_list.append(fp)
    return file_path_list

# 保存路径到txt
def save_listtxt(file_path_list, save_path):
    with open(save_path, "w") as f:
        for fp in file_path_list:
            f.write(fp)
            f.write("\n")

# 分割训练集和验证集
def segment_dataset(train_ratio, file_path_list):
    train_path_list = file_path_list[:int(len(file_path_list)*train_ratio)]
    val_path_list = file_path_list[int(len(file_path_list)*train_ratio):]
    print("Num of Train: %d, Num of Val: %d" % (len(train_path_list), len(val_path_list)))
    return train_path_list, val_path_list

# 合并pickle数据
def merge_pickle(file_path):
    # load list of pickle file path
    fpl = []
    for fn in os.listdir(file_path):
        fp = os.path.join(file_path, fn)
        fpl.append(fp)
    # load pickle data in to new list
    ppg_sequences = []
    acoustic_sequences = []
    for fp in fpl:
        print('Loading data from %s.' % fp)
        with open(fp, 'rb') as f:
            data = pickle.load(f)
            ppg_sequences.extend(data[0])
            acoustic_sequences.extend(data[1])
    # save the new pickle data
    merge_file_name = os.path.join(file_path, "merge.txt") 
    print('Caching data to %s.' % merge_file_name)
    with open(merge_file_name, 'wb+') as f:
        pickle.dump([ppg_sequences, acoustic_sequences], f) 


class Audio:
    """音频对象"""
    def __init__(self, audio_path=""):
        """读取path"""
        self.path = audio_path
        self.get_audio_from_path()

    def get_audio_from_path(self):
        """运行获取来自文件的音频数据"""
        if self.path == "":
            print("Please call the get_audio_from_data(data) or input a path by calling the get_path(self, path)!")
        else:
            f = self.__get_sr__()
            self.au, self.sr = librosa.load(self.path, sr=f) 
        
    def audio_detail(self):
        '''
        查看音频属性, sr...
        print 声道，采样宽度，帧速率，帧数，唯一标识，无损
        '''
        print('---------声音信息------------')
        with wave.open(self.path, 'rb') as f:
            for item in enumerate(f.getparams()):
                print(item)

    def plot_waveform(self):
        """画出波形"""
        # self.get_audio_from_path()
        time = self.au.shape[0] / self.sr
        st = 1 / self.sr
        x_seq = np.arange(0, time, st)

        plt.plot(x_seq, self.au, 'blue')
        plt.xlabel("time (s)")
        plt.show()

    def return_audio_data(self):
        try:
            return self.au, self.sr
        except AttributeError:
            print("Please load audio data!")
    
    def get_path(self, path):
        """传入path"""
        self.path = path
        print("Load audio from the path: ", self.path)

    def play_audio(self):
        """jupyter用"""
        ipd.Audio(self.path)

    def get_audio_from_data(self, data, sr):
        """
        好像一般不会用到
        运行获取来自数组的音频数据
        data: Audio data, type: 1-D nparray
        sr: Sample rate, type: int
        """
        self.au = data
        self.sr = sr

    def __get_sr__(self):
        """获得采样率的参数"""
        file = wave.open(self.path)
        f = file.getparams().framerate  # 采样频率 
        return f



# ------------ PPGs ----------------- #
# ----------------------------------- #
PHONEMES = ["aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er", "ey", "f", "g", "hh", "ih", 
            "iy", "jh", "k", "l", "m", "n", "ng", "ow", "oy", "p", "r", 
            "s", "sh", "t", "th", "uh", "uw", "v", "w", "y", "z", "zh", "sil"]
# Get PPGs
def get_ppg(wav_path, deps):
    """
    Input:
        wav_path: *.wav path
        deps = DependenciesPPG()
    Library:
        from scipy.io import wavfile
        from common import feat
    """
    fs, wav = wavfile.read(wav_path)
    wave_data = feat.read_wav_kaldi_internal(wav, fs)
    seq = ppg.compute_full_ppg_wrapper(wave_data, deps.nnet, deps.lda, 10)
    return seq

# Plot 39dim PPGs
def image_ppg(ppg_np):
    """
    Input: 
        ppg: numpy array
    Return:
        ax: 画布信息
        im：图像信息
    """
    ppg_deps = ppg.DependenciesPPG()
    ppg_M = Matrix(ppg_np)
    monophone_ppgs = ppg.reduce_ppg_dim(ppg_M, ppg_deps.monophone_trans)
    monophone_ppgs = monophone_ppgs.numpy().T

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(monophone_ppgs, aspect="auto", origin="lower",
                   interpolation='none')
    return ax, im

def plot_ppg(ax, im):
    ax.set_yticks(list(range(0, 40)))
    ax.set_yticklabels(PHONEMES)
    plt.colorbar(im, ax=ax)
    plt.ylabel("Phonemes")
    plt.xlabel("Time")
    # fig.suptitle('matplotlib.axes.Axes.set_yticklabels() function Example\n\n', fontweight ="bold")
    plt.tight_layout()
    plt.show()  

# ------------ P2M ------------------ #
# ----------------------------------- #
# data to GPU
def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

# Predict Function
def get_inference(seq, model, is_clip=False):
    """Tacotron inference.

    Args:
        seq: T*D numpy array.
        model: Tacotron model.
        is_clip: Set to True to avoid the artifacts at the end.

    Returns:
        synthesized mels.
    """
    # (T, D) numpy -> (1, D, T) cpu tensor
    seq = torch.from_numpy(seq).float().transpose(0, 1).unsqueeze(0)
    # cpu tensor -> gpu tensor
    seq = to_gpu(seq)
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(seq)
    if is_clip:
        return mel_outputs_postnet[:, :, 10:(seq.size(2)-10)]
    else:
        return mel_outputs_postnet

# Load p2m model for predict the Mel from the PPG
def load_p2m_model(checkpoint_path):   
    """
    Load PPG2Mel pre-train model
    Library:
        from common.hparams import create_hparams_stage
        from script.train_ppg2mel_new import load_model
    Input:
        p2m's checkpoint_path
    """
    hparams = create_hparams_stage()

    tacotron_model = load_model(hparams)
    tacotron_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = tacotron_model.eval()
    return tacotron_model

# Normalization for Mel
def MaxMinNormalization(X, Max, Min):
    X_n = (X - Min) / (Max - Min)
    return X_n

# Preprocess mel data for plot
def preprocess_mel_for_p2m_plot(path, tacotron_model):
    '''
    mel from model

    Library:
        from ppg import DependenciesPPG
    '''
    is_cal = 1
    if is_cal:
        deps = DependenciesPPG()
        teacher_utt_path = path
        teacher_ppg_cal = get_ppg(teacher_utt_path, deps)
    else:
        # 从磁盘读入
        pass
    mel_ac_cal = get_inference(teacher_ppg_cal, tacotron_model)
    mel_ac_cal_S_0 = mel_ac_cal.cpu().detach().numpy()[0]
    mel_ac_cal_tS_0 = mel_ac_cal_S_0[::-1]
    
    model_mel = MaxMinNormalization(mel_ac_cal_tS_0, np.max(mel_ac_cal_tS_0), np.min(mel_ac_cal_tS_0))
    print("Shape of model mel:", np.shape(model_mel))
    
    '''mel from STFT'''
    mel_au_cal, sr = librosa.load(teacher_utt_path, sr=16000)
    mel_au_cal_S_1 = librosa.feature.melspectrogram(mel_au_cal, sr=sr, n_fft=1024, hop_length=160, n_mels=80)
    mel_au_cal_logS_1 = librosa.power_to_db(abs(mel_au_cal_S_1[::-1]))
    
    stft_mel = MaxMinNormalization(mel_au_cal_logS_1, np.max(mel_au_cal_logS_1), np.min(mel_au_cal_logS_1))
    print("Shape of stft mel:", np.shape(stft_mel))
    
    return model_mel, stft_mel

# Plot spectrogram with <model_syn_mel, origin_stft_mel>
def draw_mel_pair(mel_list, path):
    """
    plot the Mel fig for model-mel and origin-mel
    Library:
        import seaborn as sns
    """

    fig, ax = plt.subplots(2,1, figsize=(15, 9), sharex=True) 
    titles = ["Model Mel", "STFT Mel"]
    for i, mel in enumerate(mel_list):
        p = sns.heatmap(mel, xticklabels=50, yticklabels=20, cmap='viridis', ax=ax[i]) 
        p.set_yticklabels([80,60,40,20])
        ax[i].set_title(titles[i] + " for " + "<" + path.split("data/")[1] + ">") 
        ax[i].set_ylabel("Channel")
        ax[i].set_xlabel("Frame")

# ----------------------------------- #
# Execute process
def main_for_p2m(checkpoint_path, path):
    # Load Model
    tacotron_model = load_p2m_model(checkpoint_path)
    # Predict model_Mel and Caculate STFT_Mel
    mel_list = preprocess_mel_for_p2m_plot(path, tacotron_model)
    # plot mel pair
    draw_mel_pair(mel_list, path)


# ------------ M2W ------------------ #
# ----------------------------------- #
# STFT

MAX_WAV_VALUE = 32768.0
waveglow_sigma = 0.6
denoiser_strength = 0.005

# ------------ Data for M2W ----------------- #
# Loads wavdata into torch array
def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = wavfile.read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

# Get Mel from path for m2w
def get_mel_for_test(file_path):
    """
    Library:
        from common.layers import TacotronSTFT
    """
    audio, sr = load_wav_to_torch(file_path)
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    stft = TacotronSTFT(filter_length=1024,
                                hop_length=160,
                                win_length=1024,
                                sampling_rate=sr,
                                mel_fmin=0.0, mel_fmax=8000.0)
    melspec = stft.mel_spectrogram(audio_norm)
    return melspec

# Load mel from model for m2w
def preprocess_model_mel_for_m2w(path, tacotron_model):
    '''mel from model'''
    deps = DependenciesPPG()
    teacher_utt_path = path
    teacher_ppg_cal = get_ppg(teacher_utt_path, deps)
    mel_ac_cal = get_inference(teacher_ppg_cal, tacotron_model)
    return mel_ac_cal

# Load model
def load_waveglow_model(path):
    model = torch.load(path)['model']
    model = model.remove_weightnorm(model)
    model.cuda().eval()
    return model

# Mel to Wave
def waveglow_audio(mel, waveglow, sigma, is_cuda_output=False):
    mel = torch.autograd.Variable(mel.cuda())
    if not is_cuda_output:
        with torch.no_grad():
            audio = 32768 * waveglow.infer(mel, sigma=sigma)[0]
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
    else:
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma).cuda()
    return audio

# Denoiser
def load_denoiser(waveglow_path):
    """
    Library:
        from waveglow.denoiser import Denoiser
    """
    waveglow_for_denoiser = torch.load(waveglow_path)['model']
    waveglow_for_denoiser.cuda()
    denoiser_mode = 'zeros'
    denoiser = Denoiser(waveglow_for_denoiser, mode=denoiser_mode)
    return denoiser


# if __name__ == "__main__":

    
