##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import torch
import copy
import logging
from utils.utils import create_logger

from TSPTrainer import TSPTrainer as Trainer
from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold
from TSPModel import TSPModel as Model

##########################################################################################
# parameters not to change
device = torch.device(f"cuda:{CUDA_DEVICE_NUM}" if USE_CUDA else "cpu")

# parameters  to change
num_clients = 5
num_rounds = 10

env_params = {
    'problem_size': 20,
    'pomo_size': 20,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [501,], #这里获取要改一下scheduler机制，因为在FedPOMO中应该在total_epochs = epochs × num_rounds = 501 时调整学习率，而不是单独的 epochs 计数达到 501
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 2,
    'train_episodes': 10,
    'train_batch_size': 4,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to load.

    }
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################

def initialize_pomo_model(model_params):
    model = Model(**model_params).to(device)
    return model




def save_init_model(model, save_path):
    filename = 'initModel.pt'
    save_file_path = os.path.join(save_path, filename)
    # 保存模型状态字典
    torch.save(model.state_dict(), save_file_path)





def main():
    model = initialize_pomo_model(model_params)
    save_path = "/home/zhouzhiyan-uestc/workspace/POMO/NEW_py_ver/TSP/FedPOMO/result/20231222_125445_test__tsp_n20"
    save_init_model(model, save_path)



##########################################################################################

if __name__ == "__main__":
    main()


