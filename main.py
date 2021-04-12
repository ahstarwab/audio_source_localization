# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
import argparse
import torch
import torch.nn as nn
import pdb
import argparse
import yaml 
import numpy as np
from torch.utils.data import DataLoader
import os
import pickle
from pathlib import Path
from trainer import ModelTrainer, ModelTester
from utils.setup import setup_solver
from utils.loss import create_criterion

# def pickle_50():
#     pkl_p = Path('/home/nas/user/sanghoon/lastyear50/')
#     list_ = [str(x) for x in pkl_p.iterdir() if x.sufix == '.wav']
    
#     with open('data/t3_gt.json') as f:
#         data = json.load(f)    

#     for item_ in data['track3_results']:
#         id_, angle = item_['id'], item_['angle']
#         name = 'enhanced/t3_audio_{:04d}'.format(id_)
#         self.datalist.append((name, angle))



#     with open('')



def tr_val_split(data_path):
    seed = 5
    pkl_list = np.array([str(x) for x in Path(data_path).iterdir() if x.suffix == '.pkl'])
    np.random.seed(seed)
    split_ratio = 0.9
    data_len = len(pkl_list)
    idx = np.arange(data_len)
    np.random.shuffle(idx)
    # Erase two lines
    # idx = idx[:10000]
    # data_len = 10000
    split_idx = int(split_ratio*data_len)
    return pkl_list[idx], pkl_list[idx[split_idx:]]


def add_fitting_data():
    pkl_list = np.array([str(x) for x in Path('/home/nas/user/minseok/ai_50').iterdir() if x.suffix == '.pkl'])
    return pkl_list



def train(config):
    from train_data import Audio_Reader, Audio_Collate

    train_list, val_list = tr_val_split(config['datasets']['enhanced'])
    tmp_list = add_fitting_data()
    train_list = np.concatenate([train_list, tmp_list], axis=0)

    '''Data loader'''
    train_dataset = Audio_Reader(train_list)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: Audio_Collate(x), num_workers=0)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: Audio_Collate(x), num_workers=0)
    valid_dataset = Audio_Reader(val_list)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: Audio_Collate(x), num_workers=50)
    
    from model import JUNGMIN
    localizer = JUNGMIN()

    criterion = create_criterion(config['criterion']['name'])
    optimizer, scheduler = setup_solver(localizer.parameters(), config)
    
    print("Mode : Train")
    trainer = ModelTrainer(localizer, train_loader, valid_loader, criterion, optimizer, scheduler, config, **config['trainer'])
    trainer.train()

def test(config):
    from test_data_pkl import Test_Reader
    test_dataset = Test_Reader(config['datasets']['test'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory = True, num_workers=0)
    print("Mode : Test")

    #from model import pretrained_Gated_CRNN8
    #localizer = pretrained_Gated_CRNN8(10)
    from model import JUNGMIN
    localizer = JUNGMIN()

    tester = ModelTester(localizer, test_loader, config['tester']['ckpt_path'], config['tester']['device'])
    tester.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.', help='Root directory')
    parser.add_argument('-c', '--config', type=str, help='Path to option YAML file.')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--mode', type=str, help='Train or Test')
    args = parser.parse_args()


    '''Load Config'''
    with open(os.path.join(args.config, args.dataset + '.yml'), mode='r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    if args.mode == 'Train':
        train(config)
    elif args.mode == 'Test':
        test(config)
