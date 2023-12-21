
import torch
import numpy as np

#生成 batch_size 个TSP实例，即 batch_size 张不同布局的图。每个实例具有 problem_size 个节点，这些节点在二维空间中随机分布。
def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
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