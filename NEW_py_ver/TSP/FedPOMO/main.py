"""
每个客户端的分布形式均不同,不再限制last_trainer
对每个客户端都进行绘图
"""
##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 2


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
from utils.utils import *

from TSPTrainer import TSPTrainer as Trainer
from TSPModel import TSPModel as Model

from plots_train_saved import *

##########################################################################################
# parameters not to change
device = torch.device(f"cuda:{CUDA_DEVICE_NUM}" if USE_CUDA else "cpu")

# parameters to change
num_clients = 3 # 更改num_clients时也需要更改client_env_params
num_rounds = 20

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
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
        'milestones': [3001,], #对于tsp20和tsp50是501，对于tsp100是3001。这里获取要改一下scheduler机制，因为在FedPOMO中应该在total_epochs = epochs × num_rounds = 501 时调整学习率，而不是单独的 epochs 计数达到 501
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 20,
    'train_episodes': 100*1000,
    'train_batch_size': 64,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        #'path': './result/20231227_222700_train__tsp_n50',  # directory path of pre-trained model and log files saved.
        #'epoch': 10,  # epoch version of pre-trained model to load.

    }
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n100',
        'filename': 'run_log'
    }
}

##########################################################################################

def initialize_pomo_model(model_params):
    model = Model(**model_params).to(device)
    return model


def fed_avg(models):
    #接收模型对象列表，并返回一个更新后的全局模型对象。注意在federated_train函数中是先对模型状态字典进行操作，最后将模型字典转换为对象传输给fedavg函数。
    global_state_dict = {key: torch.mean(torch.stack([model.state_dict()[key] for model in models]), dim=0) 
                         for key in models[0].state_dict()}
    
    global_model = Model(**model_params).to(device)
    global_model.load_state_dict(global_state_dict)
    
    return global_model


def federated_train(num_clients, global_model, env_params, model_params, optimizer_params, trainer_params, num_rounds):
    client_models = [None for _ in range(num_clients)]  # 初始化客户端模型列表

    # 初始化每个客户端的优化器状态字典
    client_optimizer_states = {i: None for i in range(num_clients)}

    for round in range(num_rounds):
        logger = logging.getLogger('root')
        logger.info("=================== Communicate Round {} ========================".format(round+1))
        
        # 训练每个客户端模型
        for i in range(num_clients):
            # 根据客户端ID修改数据分布类型
            client_env_params = env_params.copy()
            if i == 0:
                client_env_params['distribution'] = 'gaussian'
            elif i == 1:
                client_env_params['distribution'] = 'cluster'
            else:
                client_env_params['distribution'] = 'uniform'

            # 创建新的 Trainer 实例
            trainer = Trainer(env_params=client_env_params,
                              model_params=model_params,
                              optimizer_params=optimizer_params,
                              trainer_params=trainer_params,
                              client_num=i+1)
            
            trainer.logger.info(" *** Client{}:{} Start Training *** ".format(i+1, client_env_params['distribution']))

                
            ##global_model在外循环中更新，所以通信频率就是内循环结束，每个client_model训练完所设置的epochs次数。
            trainer.model.load_state_dict(copy.deepcopy(global_model.state_dict())) 

            # 如果该客户端有保存的优化器状态，则加载它
            if client_optimizer_states[i]:
                trainer.optimizer.load_state_dict(client_optimizer_states[i])
            
            trainer.run()

            # 更新客户端模型状态字典
            # 注意client_models列表现在包含的是模型的状态字典（一个OrderedDict对象），而不是模型对象本身
            client_models[i] = copy.deepcopy(trainer.model.state_dict())
            client_optimizer_states[i] = trainer.optimizer.state_dict()


        # 创建模型对象列表，用于聚合
        client_models_objs = [Model(**model_params).to(device) for _ in range(num_clients)]
        for model_obj, state_dict in zip(client_models_objs, client_models):
            model_obj.load_state_dict(state_dict)

        # 聚合模型参数
        global_model = fed_avg(client_models_objs)

         # 每一轮通信都保存全局模型和其他结果
        save_global_model(global_model, trainer.result_folder, round)

    return global_model


def save_global_model(model, save_path, round):
    #保存全局模型到指定路径。
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 修改文件名以包含round参数。由于round从0开始，所以加1。
    filename = f'globalModel-{round + 1}.pt'
    save_file_path = os.path.join(save_path, filename)
    # 保存模型状态字典
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
    logger.info('num_clients: {}, num_rounds: {}'.format(num_clients, num_rounds))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    # 初始化global_model
    global_model = initialize_pomo_model(model_params) 

    # 调用联邦学习训练函数
    global_model = federated_train(num_clients, global_model, env_params, model_params, optimizer_params, trainer_params, num_rounds)  

    # 绘制各个客户端训练的图像
    result_folder = get_result_folder()
    run_log_path = '{}/run_log'.format(result_folder)
    client_data = parse_log_data(run_log_path)
    plot_scores_and_losses(client_data, result_folder)
    logger = logging.getLogger('root')
    logger.info("Plots are saved successfully.")


##########################################################################################

if __name__ == "__main__":
    main()


