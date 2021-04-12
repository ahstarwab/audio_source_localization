import json
import sys
sys.path.append('..')
from torch.utils.data.dataset import Dataset
from pathlib import Path
import pickle
import pdb
import torch
import numpy as np
import argparse
import os
import sys
import librosa
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def Audio_Collate(batch):
    data, angles = list(zip(*batch))
    data_len = torch.LongTensor(np.array([x.size(1) for x in data if x.size(1)!=1]))
    if len(data_len) == 0:
        return -1
    max_len = max(data_len)
    wrong_indices = []
    
    for i, a_ in enumerate(angles):
        if a_[0] == -1:
            wrong_indices.append(i)
    B = len(data)
    inputs = torch.zeros(B-len(wrong_indices), 6, max_len, 257)
    labels = torch.zeros(B-len(wrong_indices), 10)
    
    j = 0

    '''zero pad'''    
    # for i in range(B):
    #     if i in wrong_indices:
    #         continue

    #     inputs[j, :, :data[i].size(1),:] = data[i]
    #     labels[j, angles[i]] = 1.0
    #     j += 1


    '''replica'''
    for i in range(B):
        if i in wrong_indices:
            continue

        inputs[j, :, :data[i].size(1),:] = data[i]
        labels[j, angles[i]] = 1.0

        num_pad = max_len - data[i].size(1) # To be padded
        idx_ = data[i].size(1)
        while num_pad > 0:
            if num_pad > data[i].size(1):
                inputs[j, :, idx_:idx_+data[i].size(1),:] = data[i]
                idx_ += data[i].size(1)
                num_pad -= data[i].size(1)
            else:
                inputs[j, :, idx_:idx_+num_pad,:] = data[i][:,:num_pad,:]
                num_pad = 0
        j += 1

    data = (inputs, labels, data_len)
    
    return data



class Audio_Reader(Dataset):
    def __init__(self, datalist):
        super(Audio_Reader, self).__init__()
        self.datalist = datalist
        
        self.nfft = 512
        self.hopsize = self.nfft // 4
        self.window = 'hann'

    def __len__(self):
        return len(self.datalist)

    def FeatureExtractor(self, sig):
        def mag(sig):

            S = np.abs(librosa.stft(y=sig,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2        
        
            S[:10, :] = 0.0
            return S


        def vad(spec):
            
            S3 = (spec[0] + spec[1])
            indices = S3.sum(0) > S3.sum(0).mean()/50
            return np.stack([spec[0][:,indices], spec[1][:,indices]])

        def transform(audio):

            channel_num = audio.shape[0]
            feature_logmel = []
            for n in range(channel_num):
                feature_logmel.append(mag(audio[n]))

            feature_logmel = vad(feature_logmel)
            # feature_logmel = np.concatenate(feature_logmel, axis=0)
            return feature_logmel
        
        return transform(sig)

    def vad(self, audio):
        s1 = abs(audio[0]).mean()
        s2 = abs(audio[1]).mean()

        if s1 > s2:
            gizun = s1 / 50
            indices = abs(audio[0]) > gizun
        else:
            gizun = s2 / 50
            indices = abs(audio[1]) > gizun
        
        return audio[:, indices], indices


    def vad_sanghoon(self, audio, time):
        # fname = fname.replace('1_enhanced', '2_VAD')
        audio_list = []

        for t_ in time:
            start, end = t_
            audio_list.append(audio[:, int(16000*start)+512:int(16000*end)-512])
        if len(audio_list) > 1:
            return np.concatenate(audio_list, 1)
        else:
            return audio_list[0]


    def __getitem__(self, idx):

        with open(self.datalist[idx], 'rb') as f:
            data = pickle.load(f)
        
        audio_path, angle, time, LR = data['audio_path'], data['angle'], data['time'], data['LR']
        audio_path = audio_path.replace('/home/jungwook/AOSE_Unet/', './')
        audio_enhanced, _ = librosa.load(audio_path, sr=16000 , mono=False, dtype=np.float32)

        audio_path = audio_path.replace('/home/nas/DB/AI_grand_challenge_2020/jungwook_test/wind_train_wav2minsuk', '/home/nas/DB/AI_grand_challenge_2020/jungwook_wind_drone_18_20_rec')
        audio_path = audio_path.replace('db_d', 'db/d')
        audio_path = audio_path.replace('./jungwook8//enhance_wav', './data')
        audio_noisy, _ = librosa.load(audio_path, sr=16000 , mono=False, dtype=np.float32)
        
        #[C, T]

        if audio_enhanced.sum() == 0.0 or len(time) == 0:
            return torch.FloatTensor(3,1,1), np.array([-1])

        else:
            audio_enhanced = self.vad_sanghoon(audio_enhanced, time)
            audio_noisy = self.vad_sanghoon(audio_noisy, time)
            audio_enhanced, indices = self.vad(audio_enhanced)
            audio_noisy = audio_noisy[:, indices]
            feature_enhanced = self.LogMelGccExtractor(audio_enhanced)
            feature_noisy = self.LogMelGccExtractor(audio_noisy)
            
            '''(channels, seq_len, mel_bins)'''
            # pdb.set_trace()
            return torch.FloatTensor(np.concatenate([feature_enhanced, feature_noisy], axis=0)).transpose(1,2), np.array([int(x)//20 for x in angle])

            # return feature, np.array([int(x)//20 for x in angle])
