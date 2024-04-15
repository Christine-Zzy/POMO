
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module): #是整个模型的主要类。它包含了编码器（encoder）和解码器（decoder），并实现了预处理和前向传播的方法。

    def __init__(self, **model_params): #初始化模型，创建编码器和解码器的实例。encoded_nodes 是编码器对输入问题进行编码得到的节点表示。
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params) #在模型初始化时，创建了一个名为 encoder 的成员变量，它是一个 TSP_Encoder 类的实例。TSP_Encoder 是模型的编码器部分，用于将问题数据进行嵌入。TSP_Encoder 的初始化参数来自 model_params。
        self.decoder = TSP_Decoder(**model_params) #类似地，创建了一个名为 decoder 的成员变量，它是一个 TSP_Decoder 类的实例。TSP_Decoder 是模型的解码器部分，用于生成问题的解决方案。同样，TSP_Decoder 的初始化参数来自 model_params。
        self.encoded_nodes = None #是一个用于存储编码器输出的变量，初始值设置为 None。在 pre_forward 阶段，会将编码器的输出赋值给这个变量。
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state): #在每个 epoch 开始前进行预处理，通过编码器对输入问题进行编码，将编码结果存储在 encoded_nodes 中。
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM) # batch：表示批次大小，即在一个训练批次中有多少个样本。problem：表示问题的数量，用于表示批次中问题的总数量。EMBEDDING_DIM：表示嵌入维度，即每个问题或节点的嵌入向量的维度。
        self.decoder.set_kv(self.encoded_nodes)
#是模型的前向传播方法，用于计算模型的输出。它接受一个名为 state 的参数，该参数包含了当前状态信息。在前向传播中，根据当前状态判断是否为第一个动作，如果是，则直接选取所有可能的动作并赋予均等的概率，然后为解码器初始化一个信息。如果不是第一个动作，则根据解码器生成当前状态下的动作概率分布，然后根据不同的模式（训练或评估）进行采样或选择动作。
    def forward(self, state): 
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None: #如果当前状态为第一次动作
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size) #模型会选择所有可能的目标（pomo_size 个），并为每个目标赋予相同的概率。然后，模型会将第一个节点的编码用于解码器的查询 q1，以便后续的动作生成过程中能够与其进行注意力计算。
            prob = torch.ones(size=(batch_size, pomo_size))
            
            encoded_first_node = _get_encoding(self.encoded_nodes, selected) #使用 _get_encoding 函数来获取编码后的首个节点表示。
            # shape: (batch, pomo, embedding) # pomo即pomo_size，表示在解码器中同时处理的问题的数量。
            self.decoder.set_q1(encoded_first_node) #encoded_first_node 是经过编码器编码后的第一个节点的表示。在解码过程的第一个动作中，模型需要一个初始的查询向量，以便在生成第一个动作时聚焦于问题的起始状态。

        else: #如果不是第一个动作，则根据解码器生成当前状态下的动作概率分布，然后根据不同的模式（训练或评估）进行采样或选择动作。 
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if self.training or self.model_params['eval_type'] == 'softmax': #采样策略
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2) #贪心策略
                # shape: (batch, pomo)
                prob = None


        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module): #是编码器类，它对输入问题进行编码得到节点表示。
    def __init__(self, **model_params): #初始化编码器，包括输入嵌入层和多层自注意力编码层。
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data): #通过输入嵌入层将问题数据嵌入到向量表示中，然后通过多层自注意力编码层对嵌入进行进一步编码，得到最终的节点表示。
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module): #是编码器的一个注意力层，用于对节点表示进行自注意力编码。
    def __init__(self, **model_params): #初始化自注意力层，包括查询（Q）、键（K）、值（V）的线性变换层，以及残差连接和归一化层。
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)
 
    def forward(self, input1): #实现了自注意力编码的逻辑，包括计算 Q、K、V，计算多头注意力，合并多头输出并经过线性层和残差连接，最终输出经过归一化的节点表示。
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module): #是译码器类，根据输入状态和已编码的节点表示预测选择的目标概率。
    def __init__(self, **model_params): #初始化解码器，包括线性变换层和多头注意力编码层，还有一些中间变量。
        super().__init__() 
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes): #设置解码器需要的键值对。
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1): #设置解码器需要的查询.
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask): #定义了解码器的前向传播逻辑，包括多头注意力计算和单头注意力计算，以及概率计算。
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num): #将输入张量进行维度重排，用于多头注意力的计算。
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None): #实现多头注意力的计算逻辑。
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module): #实现残差连接和归一化的模块。
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module): #实现前馈神经网络的模块。
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
