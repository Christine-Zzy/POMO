
import torch

import os
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from utils.utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL #初始化环境和模型：创建了 TSPEnv 类的实例作为环境，并创建了 TSPModel 类的实例作为模型。
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        # 检查 'model_state_dict' 键是否在 checkpoint 中，并据此加载模型状态
        if 'model_state_dict' in checkpoint:
            # 如果存在 'model_state_dict' 键，则使用它的值来加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果不存在 'model_state_dict' 键，则直接使用整个检查点来加载模型状态
            self.model.load_state_dict(checkpoint)

        # utility
        self.time_estimator = TimeEstimator()

    def run(self): #这个方法用于运行测试。在每次测试循环中，它会处理一批测试样本，计算评分并记录日志。测试过程中计算了 NO-AUG 分数和 AUG 分数
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size): #用于处理一批测试样本。在进行测试之前，首先根据是否开启数据增强（augmentation）来设置增强因子 aug_factor。然后将模型设置为评估模式（eval），并用无梯度计算的上下文 torch.no_grad() 对环境和模型进行初始化。
#另外在 _test_one_batch 函数中，存在贪心选择操作，类似于 Algorithm 2 中的 GREEDYROLLOUT。在这里，模型根据当前状态进行推断，选择动作，并执行环境交互，直到完成一个 episode。使用了 reward 计算各种分数（no-aug 和 aug 分数）。
        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']: #如果启用了数据增强，就会将 aug_factor 设置为大于1的值，从而生成多倍的起始节点集合。
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor) #在加载问题时，会调用 self.env.load_problems 函数来进行数据增强，生成多个不同的起始节点集合。具体的增强操作是在 augment_xy_data_by_8_fold 函数中实现的，它将问题的 x、y 坐标进行不同的组合，生成多个不同的起始节点集合。
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout #在测试过程中，对于每个样本，使用模型进行 POMO Rollout，以获取模型的预测结果。
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation #从所有起始节点集合和路径中选取的最高奖励值
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value #然后通过取负号并计算平均值，得到 aug_score，表示在所有起始节点集合和路径中的平均奖励

        return no_aug_score.item(), aug_score.item() #根据测试的结果，计算不使用数据增强时的得分 no_aug_score 和使用数据增强时的得分 aug_score。
