import pickle
import numpy as np

from common.hparams_628 import create_hparams_stage
from script.train_ppg2mel_628 import load_model
import torch

import matplotlib.pyplot as plt
import librosa.display
import librosa

from common.hparams_628 import create_hparams_stage
from script.train_ppg2mel_628 import load_model
import torch

from ppg import DependenciesPPG
from scipy.io import wavfile
from common import feat
from common import ppg

import seaborn as sns

hparams = create_hparams_stage()
is_clip = False
deps = DependenciesPPG()

# Get PPGs
def get_ppg(wav_path, deps):
    fs, wav = wavfile.read(wav_path)
    wave_data = feat.read_wav_kaldi_internal(wav, fs)
    seq = ppg.compute_full_ppg_wrapper(wave_data, deps.nnet, deps.lda, 10)
    return seq

# data to GPU
def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def get_inference(seq, model, is_clip=False):
    # (T, D) numpy -> (1, D, T) cpu tensor
    seq = torch.from_numpy(seq).float().transpose(0, 1).unsqueeze(0)
    # cpu tensor -> gpu tensor
    seq = to_gpu(seq)
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(seq)
    if is_clip:
        return mel_outputs_postnet[:, :, 10:(seq.size(2)-10)]
    else:
        return mel_outputs_postnet

# Taco class
class Taco_Model:
    def __init__(self, model_path, hparams):
        self.path = model_path
        self.hparams = hparams
        self.model = load_model(hparams)
        self.init_model()
    
    def load_dict(self):
        self.model.load_state_dict(torch.load(self.path)['state_dict'])
        
    def eval_model(self):
        _ = self.model.eval()
    
    def init_model(self):
        self.load_dict()
        self.eval_model()

# 处理来自pickle的ppg
# PPG_via_pickle().mel
class PPG_via_pickle:
    def __init__(self, fpath, model, index=0, is_clip=False, mp=""):
        self.fpath = fpath
        self.model = model
        self.index = index
        self.is_clip = is_clip
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        self.ppg_seq = data[0]
        
        self.ppg = self.ppg_seq[self.index]
        self.mel = self.process()
        self.mp = mp
        
    def process(self):
        #print("PPG's shape: ", np.shape(self.ppg))
        ac_mel = get_inference(self.ppg, self.model, self.is_clip)
        ac_mel_np = ac_mel.cpu().detach().numpy()
        #print("Mel's shape: ", np.shape(ac_mel_np[0]))
        ac_mel_np_re = ac_mel_np[0]
        return ac_mel_np_re
        
    def plot_mel(self):
        plt.subplots(figsize=(15,3))
        plt.title("Checkpoint is " + self.mp.split(sep="checkpoint_")[1])
        sns.heatmap(self.mel[::-1], cmap='viridis')
        plt.show()

# 处理抽取自wav的ppg
# PPG_via_wav.mel / PPG_via_wav.n_mel
class PPG_via_wav:
    def __init__(self, deps, wpath, model=None, is_stft=False):
        self.wpath = wpath
        self.deps = deps
        self.ppg = get_ppg(wpath, deps)
        self.model = model
        if is_stft:
            self.mel = self.process_via_stft()
        else:
            self.mel = self.process_via_model()
        self.n_mel = self.norm_mel()

    def process_via_model(self):
        mel_ac_cal = get_inference(self.ppg, self.model, is_clip)
        mel_ac_cal_S_0 = mel_ac_cal.cpu().detach().numpy()[0]
        mel_ac_cal_tS_0 = mel_ac_cal_S_0[::-1]
        # np.shape(mel_ac_cal_tS_0)
        return mel_ac_cal_tS_0
    
    def process_via_stft(self):
        mel_au_cal, sr = librosa.load(self.wpath, sr=16000)
        mel_au_cal_S_1 = librosa.feature.melspectrogram(mel_au_cal, sr=sr, n_fft=1024, hop_length=160, n_mels=80)
        mel_au_cal_logS_1 = librosa.power_to_db(abs(mel_au_cal_S_1[::-1]))
        # np.shape(mel_au_cal_logS_1) 
        return mel_au_cal_logS_1
    
    def norm_mel(self):
        n_mel =  (self.mel-np.min(self.mel))/(np.max(self.mel)-np.min(self.mel))
        return n_mel

    def plot_mel(self, mel_type, mp="", save_path=""):
        plt.subplots(figsize=(15,3))
        plt.title("Checkpoint is " + mp.split(sep="_")[-1])
        sns.heatmap(mel_type[::-1], cmap='viridis')
        if save_path != "":
            plt.savefig(save_path+mp.split(sep="_")[-1]+"t.png")
        plt.show()



if __name__ == "__main__":
    """
    使用实例
    """
    # fp = "/home/fuu/sdb/data/cmu/ppg_mel_pair/bdl_val/file_20_1.txt"
    # mp = "/home/fuu/sdb/model/fac_base_cmu/p2m_bdl/checkpoint_75800"
    # wp = ""

    # tm = Taco_Model(mp, hparams)
    # pp = PPG_via_pickle(fp, tm.model)
    # pp.plot_mel()

    # pw_m = PPG_via_wav(deps, wp, tm.model)
    # pw_s = PPG_via_wav(deps, wp, is_stft=True)
    # pw_m.plot_mel(pw_m.n_mel)
    # pw_s.plot_mel(pw_s.n_mel[::-1])
