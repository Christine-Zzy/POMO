"""
用于自己手动指定log文件绘图,可以限定绘制epoch的数量或者全部绘制
"""
import matplotlib.pyplot as plt
import re

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
from utils.utils import *


# 解析日志文件以获取数据
def parse_log_data(log_file):
    data = {}
    current_client = ''  # 当前正在处理的客户端
    with open(log_file, 'r') as file:
        for line in file.readlines():
            client_match = re.search(r'\*\*\* (Client\d+):(\w+) Start Training \*\*\*', line)
            if client_match:
                current_client = client_match.group(1)
                client_type = client_match.group(2)
                if current_client not in data:
                    data[current_client] = {'type': client_type, 'score': [], 'loss': [], 'epoch': []}

            # 检测train_score_list和train_loss_list的行
            score_match = re.search(r'train_score_list = \[([\d\.,\s\-]+)\]', line)
            if score_match and current_client:
                scores = [float(x) for x in score_match.group(1).split(',')]
                data[current_client]['score'].extend(scores)
                data[current_client]['epoch'].extend(range(len(data[current_client]['score']) - len(scores) + 1, len(data[current_client]['score']) + 1))

            loss_match = re.search(r'train_loss_list = \[([\d\.,\s\-]+)\]', line)
            if loss_match and current_client:
                losses = [float(x) for x in loss_match.group(1).split(',')]
                data[current_client]['loss'].extend(losses)

    return data


# 绘图函数
def plot_scores_and_losses(client_data, result_folder):
    client_num = 0
    for client, values in client_data.items():
        client_num += 1

        # 确保有有效的数据可绘制
        
        if not values['score'] or not values['loss']:
            print(f"Skipping {client} due to empty score or loss.")
            continue
        if len(values['epoch']) != len(values['score']) or len(values['epoch']) != len(values['loss']):
            print(f"Data length mismatch in {client}. Check the log parsing.")
            continue

        plt.figure(figsize=(20, 10))

        # 绘制 Score 图
        plt.subplot(2, 1, 1)
        plt.plot(values['epoch'][:60], values['score'][:60], marker='o')# 限制为前几个epoch的数据
        #plt.plot(values['epoch'], values['score'], marker='o')# 所有epoch的数据
        plt.title(f"{values['type']} Train Score")
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        # 设置x轴的步长以及范围
        max_epoch = max(values['epoch'])
        plt.xticks(range(1, 61, 2))  # 绘制前x个epoch的数据，注意中间的参数=x+1
        #plt.xticks(range(1, max_epoch + 1, 3))  # 绘制所有epoch的数据

        # 绘制 Loss 图
        plt.subplot(2, 1, 2)
        plt.plot(values['epoch'][:60], values['loss'][:60], marker='o', color='red')# 限制为前几个epoch的数据
        #plt.plot(values['epoch'], values['loss'], marker='o', color='red')
        plt.title(f"{values['type']} Train Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(range(1, 61, 2))  # 绘制前x个epoch的数据，注意中间的参数=x+1

        # 保存图像
        image_prefix = os.path.join(result_folder, f'client_{client_num}')
        os.makedirs(image_prefix, exist_ok=True)
        plt.savefig(f'{image_prefix}/training_plot.png')
        plt.close()


#手动针对某run_log文件进行绘图
result_folder = '/home/zhouzhiyan-uestc/workspace/POMO/NEW_py_ver/TSP/FedPOMO/result/20240408_083242_train__tsp_n20'
run_log_path = '{}/run_log'.format(result_folder)
client_data = parse_log_data(run_log_path)
plot_scores_and_losses(client_data, result_folder)
