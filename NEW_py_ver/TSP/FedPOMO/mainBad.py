##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 4


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
from TSPEnv import TSPEnv

##########################################################################################

# parameters 
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
        'milestones': [501,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 510,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
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
# main

if USE_CUDA:
    cuda_device_num = trainer_params['cuda_device_num']
    torch.cuda.set_device(cuda_device_num)
    device = torch.device('cuda', cuda_device_num)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')


def fed_avg(models):
    global_state_dict = {key: torch.mean(torch.stack([model.state_dict()[key] for model in models]), dim=0) 
                         for key in models[0].state_dict()}
    
    global_model = Model(**model_params)
    global_model.load_state_dict(global_state_dict)
    
    return global_model


def federated_train(num_clients, global_model, env_params, model_params, optimizer_params, trainer_params, num_rounds, last_trainer=None):
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    
    for _ in range(num_rounds):
        # 训练每个客户端模型
        for i in range(num_clients):
            trainer = Trainer(env_params=env_params,
                              model_params=model_params,
                              optimizer_params=optimizer_params,
                              trainer_params=trainer_params,
                              last_trainer=last_trainer)
            

            trainer.model = client_models[i]

            trainer.run()
            client_models[i] = trainer.model

            if i == num_clients - 1: # 仅让最后一个客户端模型的训练器来输出result_log
                last_trainer = trainer

        # 每次通信时聚合一次模型参数
        global_model = fed_avg(client_models)

        # 将聚合后的全局模型参数更新到各客户端模型
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

    return global_model, last_trainer


def save_global_model(model, save_path):

    #保存全局模型到指定路径。
    #注意全局模型没有'model_state_dict' 键，客户端训练POMO时的epoch是从1开始，所以我们设定最后保存时以checkpoint-0.pt来代表全局模型，这样方便用TSPTester来测试全局模型。

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, 'checkpoint-0.pt')
    torch.save(model.state_dict(), save_file_path)


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    # 初始化模型结构
    global_model = Model(**model_params)

    global_model, last_trainer = federated_train(num_clients, global_model, env_params, model_params, optimizer_params, trainer_params, num_rounds)

    # 保存全局模型和其他结果
    save_global_model(global_model, last_trainer.result_folder)

##########################################################################################

if __name__ == "__main__":
    main()


