
import torch
import numpy as np

#生成 batch_size 个TSP实例，每个实例具有 problem_size 个节点，这些节点在二维空间中随机分布。
def get_random_problems(batch_size, problem_size, distribution='uniform'):
    if distribution == 'uniform':
        # 原始的均匀分布
        problems = torch.rand(size=(batch_size, problem_size, 2))
    elif distribution == 'gaussian': 
        # 正态分布（高斯分布），假设中心在0.5，标准差为0.1
        problems = torch.normal(mean=0.5, std=0.1, size=(batch_size, problem_size, 2))
    elif distribution == 'cluster': #使用这种分布需要限定batch_size为偶数
        # 聚类分布，创建两个聚类中心
        center1 = (torch.rand(1, 1, 2) * 0.5).expand(batch_size//2, problem_size, 2) # 第一个聚类中心在[0,0.5)范围内
        center2 = (torch.rand(1, 1, 2) * 0.5 + torch.tensor([0.5, 0.5])).expand(batch_size//2, problem_size, 2) # 第二个聚类中心在[0.5,1)范围内
        std = 0.05 * torch.ones_like(center1)
        cluster1 = torch.normal(mean=center1, std=std)
        cluster2 = torch.normal(mean=center2, std=std)
        problems = torch.cat([cluster1, cluster2], dim=0)
    elif distribution == 'mixed':
        # 混杂分布
        # 分配 batch_size，并且确保cluster的部分为偶数
        part_size = batch_size // 3
        remaining = batch_size % 3
        sizes = [part_size + 1 if i < remaining else part_size for i in range(3)]
        # 如果cluster部分不是偶数，从gaussian或uniform部分调整一个实例
        if sizes[2] % 2 != 0:
            sizes[1] += 1  # 从gaussian部分借用
            sizes[2] -= 1  # 确保cluster部分是偶数
        problems_uniform = get_random_problems(sizes[0], problem_size, distribution='uniform')
        problems_gaussian = get_random_problems(sizes[1], problem_size, distribution='gaussian')
        problems_cluster = get_random_problems(sizes[2], problem_size, distribution='cluster')
        problems = torch.cat([problems_uniform, problems_gaussian, problems_cluster], dim=0)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")
    return problems


#对给定的问题数据进行 8 倍的增强，即将每个问题在不同的坐标方向进行变换。输入参数 problems 是一个张量，形状为 (batch_size, problem_size, 2)，表示一批问题数据。在这个函数中，首先从问题数据中提取出 x 坐标和 y 坐标，然后按照不同的变换方式将 x 和 y 进行组合。最后，将所有的增强后的问题数据拼接在一起，形成一个新的张量，其形状为 (8 * batch_size, problem_size, 2)，表示经过 8 倍增强后的问题数据。
def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems