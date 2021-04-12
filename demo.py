# import sys
# sys.path.append('..')
import torch
import torch.nn as nn
import pdb
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import pickle
from pathlib import Path
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from collections import defaultdict
import time
import librosa


class Test_Reader(Dataset):
    def __init__(self, datalist):
        super(Test_Reader, self).__init__()

        self.nfft = 1024
        self.hopsize = 160
        self.mel_bins = 128
        self.fmin = 50
        self.window = 'hann'
        self.melW = librosa.filters.mel(sr=16000,
                                        n_fft=1024,
                                        n_mels=self.mel_bins,
                                        fmin=self.fmin)

        self.datalist = datalist



    def __len__(self):
        return len(self.datalist)

    def LogMelGccExtractor(self, sig):
        def logmel(sig):

            S = np.abs(librosa.stft(y=sig,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2        
            S_mel = np.dot(self.melW, S).T
            S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
            S_logmel = np.expand_dims(S_logmel, axis=0)

            return S_logmel

        def gcc_phat(sig, refsig):

            ncorr = 2*self.nfft - 1
            nfft = int(2**np.ceil(np.log2(np.abs(ncorr))))
            Px = librosa.stft(y=sig,
                            n_fft=nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window, 
                            pad_mode='reflect')
            Px_ref = librosa.stft(y=refsig,
                                n_fft=nfft,
                                hop_length=self.hopsize,
                                center=True,
                                window=self.window,
                                pad_mode='reflect')
        
            R = Px*np.conj(Px_ref)

            n_frames = R.shape[1]
            gcc_phat = []
            for i in range(n_frames):
                spec = R[:, i].flatten()
                cc = np.fft.irfft(np.exp(1.j*np.angle(spec)))
                cc = np.concatenate((cc[-self.mel_bins//2:], cc[:self.mel_bins//2]))
                gcc_phat.append(cc)
            gcc_phat = np.array(gcc_phat)
            gcc_phat = gcc_phat[None,:,:]

            return gcc_phat

        def transform(audio):

            channel_num = audio.shape[0]
            feature_logmel = []
            feature_gcc_phat = []
            for n in range(channel_num):
                feature_logmel.append(logmel(audio[n]))
                for m in range(n+1, channel_num):
                    feature_gcc_phat.append(
                        gcc_phat(sig=audio[m], refsig=audio[n]))
            
            feature_logmel = np.concatenate(feature_logmel, axis=0)
            feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
            feature = np.concatenate([feature_logmel, feature_gcc_phat])

            return feature
        
        return transform(sig)
        
    def RT_preprocessing(self, extractor, audio):

        '''This step needs to be considered'''
        # audio = audio / (np.max(np.abs(audio)) + np.finfo(np.float).eps)

        feature = extractor.transform(audio)
        '''(channels, seq_len, mel_bins)'''
        '''(channels, time, frequency)'''

        return feature

    def vad(self, audio):
        s1 = abs(audio[0]).mean()
        s2 = abs(audio[1]).mean()

        if s1 > s2:
            gizun = s1 / 30
            indices = abs(audio[0]) > gizun
        else:
            gizun = s2 / 30
            indices = abs(audio[1]) > gizun
        
        return audio[:, indices]


    def vad_sanghoon(self, audio, time):
        # fname = fname.replace('1_enhanced', '2_VAD')
        audio_list = []
        for t_ in time:
            start, end = t_
            audio_list.append(audio[:, int(16000*start)+512:int(16000*end)-512])
        return audio_list

    def __getitem__(self, idx):
        # pdb.set_trace()
        with open(self.datalist[idx], 'rb') as f:
            data = pickle.load(f)
        

        audio_path, time, LR = data['audio_path'], data['time'], data['LR']
        '''10'''
        audio, _ = librosa.load(audio_path, sr=16000 , mono=False, dtype=np.float32)


        '''50'''
        # audio_path = audio_path.replace('/home/jungwook/AOSE_Unet/zak0//enhance_wav', './enhanced')
        # audio_path = audio_path.replace('.wav', '_ch1.wav')
        # ch1, _ = librosa.load(audio_path, sr=16000 , mono=False, dtype=np.float32)
        # audio_path = audio_path.replace('_ch1.wav', '_ch2.wav')
        # ch2, _ = librosa.load(audio_path, sr=16000 , mono=False, dtype=np.float32)

        # audio = np.stack([ch1, ch2])

        #예외처리
        if audio.sum() == 0.0 or len(time) == 0:
            return LR, audio_path, LR
        if time[0] == -1:
            return LR, audio_path, LR

        audio_list = self.vad_sanghoon(audio, time)

        feature_list = []

        for a_ in audio_list:
            feature_list.append(self.LogMelGccExtractor(self.vad(a_)))
        
        return feature_list, audio_path, LR


class ModelTester:
    def __init__(self, model, test_loader, ckpt_path, device):

        self.device = torch.device('cuda:{}'.format(device))
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.load_checkpoint(ckpt_path)


    def load_checkpoint(self, ckpt):
        checkpoint = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        


    def test(self):
        self.model.eval()
        total_loss = 0.0
        
        output_list = []
        label_list = []
        correct_cnt = 0
        tot_cnt = 0
        batch_size = len(self.test_loader)
        # pdb.set_trace()
        with torch.no_grad():
            for b, batch in enumerate(self.test_loader):
                features, fname, LR = batch
                
                if len(features[0].size()) == 1:
                    # pdb.set_trace()
                    if features[0].item() == 0:
                        print(fname[0]+"------[2]")
                    else:
                        print(fname[0]+"------[7]")

                else:

                    for inputs in features:
                        cnt = 0
                        inputs = inputs.to(self.device)
                        outputs = self.model(inputs.float())
                    
                        
                        results = outputs.max(2)[1]#.mode()[0]
                        dict_ = defaultdict(int)
                        for i in results[0]:
                            dict_[i.item()] += 1
                        

                        final_ = []
                        for item_ in dict_.items():
                            key_, val_ = item_
                            if val_ >= len(results[0])//3:
                                final_.append(key_)
                                cnt += 1
                        
                        if cnt == 1:
                            
                            if key_ <= 4 and LR[0].item() == 1:
                                final_[-1] = 7
                                
                            elif key_ > 4 and LR[0].item() == 0:
                                final_[-1] = 2
                                                            
                        
                        print(fname[0]+"------", final_)
                        # pdb.set_trace()



# DATA_PATH = './2_VAD'
DATA_PATH = '/home/nas/DB/AI_grand_challenge_2020/jungwook_test/chochangi_2020audio/2_VAD'
# DATA_PATH = '/home/nas/DB/AI_grand_challenge_2020/jungwook_test/chochangi_2020audio/enhance_wav/'
# CKPT_PATH = '/home/nas/user/minseok/exp_AI/AI_whole/2/18November_1249/ckpt/26_94.0220.pt'
# CKPT_PATH = '/home/nas/user/minseok/exp_AI/AI_whole/2/18November_1249/ckpt/26_94.0220.pt'
CKPT_PATH = '/home/nas/user/minseok/FINAL1120/2/20November_0241/ckpt/24_92.0000.pt'

# CKPT_PATH = '/home/nas/user/minseok/8500_REAL_FINAL/2/20November_0546/ckpt/best.pt'
# CKPT_PATH = '/home/nas/user/minseok/AHSTARWB_NOISY/2/20November_1117/ckpt/21_91.6376.pt'
DEVICE = '1'


if __name__ == '__main__':
    from model import pretrained_CRNN8
    localizer = pretrained_CRNN8(10)
    pkl_list = [x for x in Path(DATA_PATH).iterdir() if x.suffix == '.pkl']
    pkl_list.sort()

    test_dataset = Test_Reader(pkl_list)
    
    test_loader =
     DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory = True, num_workers=0)
    tester = ModelTester(localizer, test_loader, CKPT_PATH, DEVICE)
    tester.test()