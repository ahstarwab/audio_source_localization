import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class JUNGMIN(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel = 514
        out_channel = 514

        self.layer1 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )



        self.layer2 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        
        self.fc = nn.Linear(514, 10)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x):
        
        B,M,T,F = x.size()
        # pdb.set_trace()
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x =  x.permute(0, 1, 3, 2).reshape(B, -1, T)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.transpose(1,2)
        
        # (x, _) = self.gru(x)
        return self.fc(x)

class JUNGMIN2(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel = 514
        out_channel = 514

        self.layer1 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )



        self.layer2 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        
        
        self.gru = nn.GRU(input_size=514, hidden_size=256, 
            num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(512, 10)

        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, x):
        
        B,M,T,F = x.size()
        # pdb.set_trace()
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x =  x.permute(0, 1, 3, 2).reshape(B, -1, T)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.transpose(1,2)
        
        (x, _) = self.gru(x)
        return self.fc(x)


class JUNGMIN3(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel = 514
        out_channel = 514

        self.layer1 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )



        self.layer2 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
                    nn.Linear(514, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)
        )


        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x):
        
        B,M,T,F = x.size()
        # pdb.set_trace()
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x =  x.permute(0, 1, 3, 2).reshape(B, -1, T)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.transpose(1,2)
        
        # (x, _) = self.gru(x)
        return self.fc(x)