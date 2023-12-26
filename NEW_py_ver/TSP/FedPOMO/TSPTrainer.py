
import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


class TSPTrainer:
    def __init__(self, #创建 TSPTrainer 类的实例时调用，用于初始化训练器的各个组件和参数。它会设置训练所需的环境参数、模型参数、优化器参数和训练参数等。还会初始化日志记录器、结果文件夹等，并根据是否使用 CUDA 进行相应的设备设置。
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 last_trainer):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.last_trainer = last_trainer #仅让最后一个客户端模型的训练器来存储checkpoints并绘图
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self): #是训练过程的主要函数。它会根据训练参数中的指定的训练轮数（epochs，也就是论文中的T），循环执行每个 epoch 的训练过程。在每个 epoch 中，会进行学习率衰减（如果有的话），调用 _train_one_epoch 方法进行训练，并记录训练得分和损失等信息。同时，它还会根据指定的间隔保存模型和日志图像，以及在训练完成后打印训练日志。
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1): #论文中的算法1循环
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch) #共生成epochs × train_episodes × train_batch_size个随机的TSP问题实例，每个实例的节点数为problem_size
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if self.last_trainer is not None and epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
               

                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if self.last_trainer is not None and (all_done or (epoch % model_save_interval) == 0):
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if self.last_trainer is not None and (all_done or (epoch % img_save_interval) == 0):
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if self.last_trainer is not None and all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
#执行一个 epoch 的训练过程。它会调用 _train_one_batch 方法来对批次中的问题进行训练，计算平均得分和平均损失，并在每个 epoch 结束时记录这些信息。该方法还在第一个 epoch 中，对前 10 个批次的训练信息进行日志记录。
    def _train_one_epoch(self, epoch): 

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size) #每个episode都会生成batch_size个TSP问题实例
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg
#执行一个批次（batch）的训练过程。首先，它准备训练所需的组件和数据，然后使用模型进行 POMO Rollout，计算损失和得分，进行反向传播更新模型。最后，它返回该批次的平均得分和平均损失。
    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size) #生成batch_size个TSP问题实例
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0)) #在代码开始时，prob_list 是一个 3D 张量，其大小为 (batch_size, pomo_size, 0)，其中 batch_size 是批次大小，pomo_size 是 POMO 的个数。0 维是空维度，它将在循环中动态地增长以存储每个目标的选择概率。
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step() #调用环境对象的 pre_step 方法，该方法返回当前状态（state）、奖励（reward）和完成标志（done）。它在每个 POMO Rollout 过程的开始阶段调用一次，以初始化环境状态。
        while not done: #只要完成标志 done 不为 True，就会执行 POMO Rollout 过程，循环选择动作，并记录每个目标的选择概率到 prob_list 中。这个过程会执行多个采样轨迹，每个轨迹都会在循环中更新选择概率。
            selected, prob = self.model(state) #selected 是模型预测的选取的动作（即选择哪些目标），prob 是每个目标被选择的概率。prob 的形状是 (batch, pomo)，其中每一行表示一个样本在每个 POMO 中选择每个目标的概率。
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected) #调用环境对象的 step 方法，传入模型预测的动作 selected，更新环境状态。它返回更新后的状态 state、奖励 reward 和完成标志 done。
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2) #在每个循环迭代中，将 prob 张量沿着第三个维度连接到 prob_list 中，导致 prob_list 的第三个维度是第三个维度增长为每个 POMO 中选择每个目标的概率。
        # Loss #在计算损失时，用的就是多个采样轨迹的概率信息作为基线，来计算损失，以减小策略梯度的方差。
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True) #advantage 计算了每个采样轨迹中的奖励与整体奖励均值的差异,即论文中公式3括号内的优势函数
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2) #log_prob 计算了每个采样轨迹中的目标选择概率的对数之和
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD 即论文中公式3的后半部分
        # shape: (batch, pomo)
        loss_mean = loss.mean() #结合后续的loss_mean.backward()实现了论文中的公式3，在POMO中loss大部分是负值且递增，reward也是递增，score递减

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo #从每个问题（pomo）的奖励中选择最大的奖励值（梯度上升的方法）。这是为了衡量在每个问题中获得的最佳结果。
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value #因为在TSPEnv.py中有reward = -self._get_travel_distance()，这里再取一个负号相当于得到self._get_travel_distance()

        # Step & Return
        ###############################################
        self.model.zero_grad() #将模型的梯度清零，以准备接收新一轮的梯度更新。
        loss_mean.backward() #用于计算loss_mean关于模型参数的梯度
        self.optimizer.step() #基于计算得到的梯度，使用优化器更新模型的参数。这将调整模型以使其更好地适应给定的训练数据。
        return score_mean.item(), loss_mean.item() #返回本次训练步骤的两个值。score_mean.item()表示当前步骤中最大化奖励的平均值，loss_mean.item()表示当前步骤中的损失函数的值。
    

    
 
