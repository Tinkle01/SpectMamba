import numpy as np
import torch
import time
import math
import torch.nn.functional as F
import torch.nn as nn
import mir_eval

LEN_SEG = 64



def melody_eval(ref_time, ref_freq, est_time, est_freq):
    output_eval = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)
    VR = output_eval['Voicing Recall'] * 100.0
    VFA = output_eval['Voicing False Alarm'] * 100.0
    RPA = output_eval['Raw Pitch Accuracy'] * 100.0
    RCA = output_eval['Raw Chroma Accuracy'] * 100.0
    OA = output_eval['Overall Accuracy'] * 100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr


def pred2res(pred):
    '''
    Convert the output of model to the result
    '''
    pred = np.array(pred)
    pred_freq = pred.argmax(axis=1)
    pred_freq[pred_freq > 0] = 31 * 2 ** (pred_freq[pred_freq > 0] / 60)
    return pred_freq


def pred2resG(pred):
    '''
    Convert the output of model to the result
    '''
    pred.float()
    pred[pred > 0] = 31 * 2 ** (pred[pred > 0] / 60)
    return pred.float()


def pred2resT(pred, threshold=0.1):
    '''
    Convert the output of model to the result
    '''
    pred = np.array(torch.softmax(pred, dim=1))
    pred_freq = pred.argmax(axis=1)
    pred_freq[pred.max(axis=1) <= 0.1] = 0
    pred_freq[pred_freq > 0] = 31 * 2 ** (pred_freq[pred_freq > 0] / 60)
    return pred_freq


def y2res(y):
    '''
    Convert the label to the result
    '''
    y = np.array(y)
    y[y > 0] = 31 * 2 ** (y[y > 0] / 60)
    return y


def convert_y(y):
    y[y > 0] = torch.round(torch.log2(y[y > 0] / 31) * 60)
    return y.long()


def convert_z(z):
    '''
    将频率转化为音调
    '''
    z[z > 0] = torch.round(12 * torch.log2(z[z > 0] / 440) + 49 - 2)
    return z.long()



def freq2tone(freq, noteFrequency):
    """
    将频率转换为音调索引
    """
    indices = torch.zeros(freq.shape)

    for i in range(len(noteFrequency)):
        # 找到频率大于 noteFrequency[i] 的位置
        mask = (freq > noteFrequency[i])
        indices[mask] = i

    return indices + 1


def convert_z2(z):
    """
    将频率转化为音调
    """
    standardFrequency = 440.0
    noteFrequency2 = [(standardFrequency / 32.0) * math.pow(2, (i - 9.0) / 12.0) for i in range(128)]
    noteFrequency = []
    for i in range(128):
        if i >= 23 and i < 89:
            noteFrequency.append(noteFrequency2[i])

    # 将 noteFrequency 转换为张量
    noteFrequency_tensor = torch.tensor(noteFrequency)


    z = z.cpu()
    # 使用 freq2tone 进行转换
    z[z > 0] = freq2tone(z[z > 0], noteFrequency_tensor)
    return z.long()


def load_list(path, mode):
    if mode == 'test':
        f = open(path, 'r')
    elif mode == 'train':
        f = open(path, 'r')
    else:
        raise Exception("mode must be 'test' or 'train'")
    data_list = []
    for line in f.readlines():
        data_list.append(line.strip())
    if mode == 'test':
        print("{:d} test files: ".format(len(data_list)), data_list)
    #else:
        #print("{:d} train files: ".format(len(data_list)), data_list)
    return data_list

def load_semi_data(path):
    tick = time.time()
    train_list = load_list(path, mode='train')
    X = []
    num_seg = 0
    for i in range(len(train_list)):
        print('({:d}/{:d}) Loading data: '.format(i + 1, len(train_list)), train_list[i])
        X_data = load_onlyx_data(train_list[i])

        seg = X_data.size(0)
        num_seg += seg
        X.append(X_data)

        print('({:d}/{:d})'.format(i + 1, len(train_list)), train_list[i], 'loaded: ', '{:2d} segments'.format(seg))
    print("Training data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))
    return torch.cat(X, dim=0)


def load_onlyx_data(fp, mode='train'):
    '''
    X: (N, C, F, T)
    '''
    X = np.load('data/cfp/' + fp)
    L = X.shape[2]
    num_seg = L // LEN_SEG
    X = torch.tensor(np.array([X[:, :, LEN_SEG * i:LEN_SEG * i + LEN_SEG] for i in range(num_seg)]),
                     dtype=torch.float32)

    return X[:num_seg]



def load_train_data(path):
    tick = time.time()
    train_list = load_list(path, mode='train')
    X, y, z = [], [], []
    num_seg = 0
    for i in range(len(train_list)):
        #print('({:d}/{:d}) Loading data: '.format(i + 1, len(train_list)), train_list[i])
        X_data, y_data, z_data = load_data(train_list[i])
        y_data[y_data > 320] = 320
        z_data[z_data > 65] = 65
        seg = X_data.size(0)
        num_seg += seg
        X.append(X_data)
        y.append(y_data)
        z.append(z_data)
        #print('({:d}/{:d})'.format(i + 1, len(train_list)), train_list[i], 'loaded: ', '{:2d} segments'.format(seg))
    print("Training data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))
    return torch.cat(X, dim=0), torch.cat(y, dim=0), torch.cat(z, dim=0)


def load_data(fp, mode='train'):
    '''
    X: (N, C, F, T)
    y: (N, T)
    z: (N, T)  for exam  [[12,2,3,0,7,45,....]
                          [0,0,0,0,12,43,....]]
    '''
    X = np.load('data/cfp/' + fp)
    L = X.shape[2]
    num_seg = L // LEN_SEG
    X = torch.tensor(np.array([X[:, :, LEN_SEG * i:LEN_SEG * i + LEN_SEG] for i in range(num_seg)]),
                     dtype=torch.float32)
    # X = (X[:, 1] * X[:, 2]).unsqueeze(dim=1)

    f = open('data/f0ref/' + fp.replace('.npy', '') + '.txt')
    y = []
    z = []
    for line in f.readlines():
        y.append(float(line.strip().split()[1]))
    num_seg = min(len(y) // LEN_SEG, num_seg)  # 防止X比y长
    y = torch.tensor(np.array([y[LEN_SEG * i:LEN_SEG * i + LEN_SEG] for i in range(num_seg)]), dtype=torch.float32)
    if mode == 'train':
        p = y.clone()
        z = convert_z2(p)
        y = convert_y(y)

    return X[:num_seg], y[:num_seg], z[:num_seg]


def f02img(y):
    N = y.size(0)
    img = torch.zeros([N, 321, LEN_SEG], dtype=torch.float32)
    for i in range(N):
        img[i, y[i], torch.arange(LEN_SEG)] = 1
    return img


def pitch2img(z):
    N = z.size(0)
    img = torch.zeros([N, 66, LEN_SEG], dtype=torch.float32)
    for i in range(N):
        img[i, z[i], torch.arange(LEN_SEG)] = 1
    return img


def pos_weight(data):
    N = data.size(0)
    non_melody = torch.sum(data[:, 0, :]).item() + 1
    melody = (N * LEN_SEG) - non_melody + 2
    z = torch.zeros((321, LEN_SEG), dtype=torch.float32)

    z[1:, :] += non_melody / melody
    z[0, :] += melody / non_melody
    return z


def ce_weight(data):
    N = data.size(0)
    non_melody = torch.sum(data == 0) + 1
    melody = (N * LEN_SEG) - non_melody + 2
    z = torch.zeros(321, dtype=torch.float32)
    z[1:] += non_melody / melody
    z[0] += melody / non_melody
    return z

def DSL(pred_u_self, eta=0.95,max_class_num=15):
    with torch.no_grad():
        # 初始化 DSL 结果
        dsl_y_star = torch.zeros_like(pred_u_self)

        # 获取所有帧的预测概率
        pt = pred_u_self.permute(2, 0, 1)  # 形状变为 (num_frames, bs, num_classes)

        # 按概率降序排序
        sorted_probs, sorted_indices = torch.sort(pt, descending=True, dim=2)
        # 计算累计概率，找到阈值类别
        accumulated_probs = torch.cumsum(sorted_probs, dim=2)
        C_star = (accumulated_probs >= eta).int().argmax(dim=2, keepdim=True)
        delta_j = sorted_probs.gather(2, C_star)
        # 只取前15个主导类别的概率
        sorted_probs, sorted_indices = (
            sorted_probs[:, :, :max_class_num],
            sorted_indices[:, :, :max_class_num],
        )
        dominant_classes = sorted_indices * (sorted_probs >= delta_j).int()
        # 创建主导类别掩码
        h_j = torch.zeros_like(pt)
        h_j.scatter_(2, dominant_classes, 1)

        # 计算动态软标签 (DSL),hj说明哪些类别是主导类别
        # 只取主导类别的概率
        y_star = h_j * pt
        y_star /= torch.sum(y_star, dim=2, keepdim=True) + 1e-10

        # 保存结果
        dsl_y_star = y_star.permute(1, 2, 0)  # 形状变回 (bs, num_classes, num_frames)

        return dsl_y_star


def calculate_cnr_factor(pred_u_self, pred_u_ema, global_confidence, m,max_class_num=15):
    #with torch.no_grad():
        # 获取所有帧的预测概率
        pt = pred_u_self.permute(2, 0, 1)  # 形状变为 (num_frames, bs, num_classes)
        pt_ema = pred_u_ema.permute(2, 0, 1)
        # 按概率降序排序
        sorted_probs, sorted_indices = torch.sort(pt, descending=True, dim=2)
        sorted_probs_ema, sorted_indices_ema = torch.sort(
            pt_ema, descending=True, dim=2
        )

        # 计算累计概率，找到阈值类别
        accumulated_probs = torch.cumsum(sorted_probs, dim=2) #(frame, bs, 321)
        local_confidence = m*torch.mean(accumulated_probs[:,:,0]) + (1-m)*global_confidence
        C_star = (accumulated_probs >= local_confidence).int().argmax(dim=2, keepdim=True)
        accumulated_probs_ema = torch.cumsum(sorted_probs_ema, dim=2)
        C_star_ema = (accumulated_probs_ema >= local_confidence).int().argmax(dim=2, keepdim=True)
        delta_j = sorted_probs.gather(2, C_star)
        delta_j_ema = sorted_probs_ema.gather(2, C_star_ema)
        # 获取主导类别
        sorted_probs, sorted_indices = (
            sorted_probs[:, :, :max_class_num],
            sorted_indices[:, :, :max_class_num],
        )
        sorted_probs_ema, sorted_indices_ema = (
            sorted_probs_ema[:, :, :max_class_num],
            sorted_indices_ema[:, :, :max_class_num],
        )
        dominant_classes = sorted_indices * (sorted_probs >= delta_j).int()
        dominant_classes_ema = (
            sorted_indices_ema * (sorted_probs_ema >= delta_j_ema).int()
        )
        # 创建主导类别掩码
        h_j = torch.zeros_like(pt)
        h_j.scatter_(2, dominant_classes, 1)
        h_j_ema = torch.zeros_like(pt_ema)
        h_j_ema.scatter_(2, dominant_classes_ema, 1)
        # 只取主导类别的概率
        y_star = h_j * pt
        y_star_ema = h_j_ema * pt_ema
        dominant_prob_sum = torch.sum(y_star, dim=2) #(64,bs)
        other_prob_sum = torch.sum(pt, dim=2) - dominant_prob_sum
        dominant_prob_sum_ema = torch.sum(y_star_ema, dim=2)
        other_prob_sum_ema = torch.sum(pt_ema, dim=2) - dominant_prob_sum_ema


        dominant_prob_sum = dominant_prob_sum.permute(1,0).unsqueeze(dim=2)
        other_prob_sum = other_prob_sum.permute(1,0).unsqueeze(dim=2)
        dominant_prob_sum_ema = dominant_prob_sum_ema.permute(1, 0).unsqueeze(dim=2)
        other_prob_sum_ema = other_prob_sum_ema.permute(1, 0).unsqueeze(dim=2)

        return (
            torch.cat([dominant_prob_sum, other_prob_sum], dim=2), #(64,2*bs)
            torch.cat([dominant_prob_sum_ema, other_prob_sum_ema], dim=2),
        )

def load_unlabeled_data(path):
    tick = time.time()
    train_list = load_list(path, mode="train")
    X = []
    num_seg = 0
    for i in range(len(train_list)):
        print(
            "({:d}/{:d}) Loading data: ".format(i + 1, len(train_list)), train_list[i]
        )
        X_data = np.load("data/cfp/" + train_list[i])
        L = X_data.shape[2]
        X_data = torch.tensor(
            np.array(
                [
                    X_data[:, :, LEN_SEG * i : LEN_SEG * i + LEN_SEG]
                    for i in range(L // LEN_SEG)
                ]
            ),
            dtype=torch.float32,
        )
        seg = X_data.size(0)
        num_seg += seg
        X.append(X_data)
        print(
            "({:d}/{:d})".format(i + 1, len(train_list)),
            train_list[i],
            "loaded: ",
            "{:2d} segments".format(seg),
        )
    print(
        "Training data loaded in {:.2f}(s): {:d} segments".format(
            time.time() - tick, num_seg
        )
    )
    return torch.cat(X, dim=0)


def contrastive_loss4(query1, predition1, query2, predition2, weights, linear_wieght, temperature=0.5, m=0.9):
    batch_size = query1.shape[0]
    query_num = query1.shape[2] # (batch_size, 321, query_num)
    pred1 = torch.softmax(predition1,dim=1)
    pred2 = torch.softmax(predition2, dim=1)

    # 扩展维度以适配批量计算
    query1 = query1.permute(0, 2, 1)  # (batch_size, query_num, 321)
    query2 = query2.permute(0, 2, 1)  # (batch_size, query_num, 321)

    # 计算正样本相似度 一行是query1对于每一个query2的相似度
    pos_sim = F.cosine_similarity(query1.unsqueeze(2), query2.unsqueeze(1), dim=3)  # (batch_size, query_num, query_num)

    mask = torch.eye(query_num, device=pos_sim.device).unsqueeze(0).expand(batch_size, -1,
                                                                           -1)  # (batch_size, query_num, query_num)
    pos_sim_diag = pos_sim * mask  # (batch_size, query_num, query_num)

    # 计算负样本相似度
    neg_sim = pos_sim.clone() *(1-mask)  # 反转单位矩阵

    #torch.fill_diagonal_(neg_sim, -float('inf'))  # 确保 i != j 的情况

    # 计算正样本置信度和最大置信度
    positive_confid = pred2.permute(0,2,1)  # (batch_size, query_num, 321)
    tch_value, tch_index = torch.max(positive_confid, dim=2)  # (batch_size, query_num)
    #print(tch_value)

    # 计算学生置信度
    anchor_confid = pred1.permute(0,2,1)  # (batch_size, query_num, 321)
    # 创建索引
    batch_indices = torch.arange(batch_size).unsqueeze(1)  # (batch_size, 1)
    query_indices = torch.arange(query_num).unsqueeze(0)  # (1, query_num)

    # 使用高级索引从 a 中提取值
    stu_value = anchor_confid[batch_indices, query_indices, tch_index] # (batch_size, query_num)

    update_weight = torch.sum(tch_value) / (batch_size * query_num)
    #print(update_weight)

    #对阈值全局调整

    global_weight = m * weights + (1-m) * update_weight  # 这里应该是应用一个ema去更新参数 （321， dim）
    with torch.no_grad():
        output = global_weight.clone()

    # 接下来对阈值局部调整
    l2_norms = torch.norm(linear_wieght, dim=1) #（321）

    l2_norms_max = torch.max(l2_norms)


    # 每个元素除以L2范数的和
    normalized_l2_norms = l2_norms / l2_norms_max
    #print(torch.max(normalized_l2_norms,dim=0))
    global_weight = global_weight.cuda() * normalized_l2_norms.cuda()



    # 计算有效正样本相似度和负样本相似度
    valid_mask = (stu_value <= tch_value) #& (tch_value >= threshold)  # (batch_size, query_num)
    consist_mask = torch.ones(batch_size, query_num)


    list2=[]
    for bs in range(batch_size):
        list1 = []
        q = tch_index[bs,:]   #每一个batch 教师的预测索引(64)
        qq = tch_value[bs, :]
        query_mask = (qq >= global_weight[q])
        valid_mask[bs] = valid_mask[bs] * query_mask
        consist_mask[bs] = query_mask
        for index in range(query_num):
            v = q[index]  #每一个batch 每一个时间 帧教师的预测索引(1)
            label = (q != v).int()
            list1.append(label.unsqueeze(0))
        list2.append(torch.cat(list1, dim=0).unsqueeze(dim=0))
    mask2 = torch.cat(list2, dim=0)

    neg_sim = neg_sim * mask2
    pos_sim = pos_sim_diag * valid_mask.unsqueeze(2).expand(-1, -1, query_num)  # 应用掩码



    # 计算损失
    pos_sim = pos_sim.sum(dim=2)  # (batch_size, query_num)
    neg_sim = neg_sim.sum(dim=2)  # (batch_size, query_num)
    loss = 0
    for bs in range(batch_size):
        x = pos_sim[bs, :]
        y = neg_sim[bs, :]
        labels = (x != 0).int()
        x = x[x != 0]
        y = y * labels
        y = y[y != 0]

        if x.shape != torch.Size([0]) and y.shape != torch.Size([0]):
            x[x<0] = x[x<0] = 0
            y[y< 0] = y[y < 0] = 0
            loss1 = -torch.log(x + 1e-10 / (x + y + 1e-10))
            loss += loss1.mean()
        else:
            continue



    #loss = -torch.log(pos_sim + 1e-2/ (pos_sim + neg_sim + 1e-2))  # (batch_size, query_num)
    loss = loss / batch_size  # 平均损失
    with torch.no_grad():
        hold_num = torch.sum(consist_mask)

    consist_mask = consist_mask.bool()




    loss2=0
    for bs in range(batch_size):
        p1 = predition1[bs,:, consist_mask[bs]].unsqueeze(dim=0)
        p2 = predition2[bs,:, consist_mask[bs]].unsqueeze(dim=0)
        #loss2 += consist_loss2(predition1[bs,:, consist_mask[bs]].unsqueeze(dim=0), predition2[bs,:, consist_mask[bs]].unsqueeze(dim=0))
        if p1.size(2) != 0:
            loss2 += consist_loss2(p1,p2)




    loss2 = loss2/batch_size


    return loss+loss2, output, hold_num

def consist_loss2(pred1, pred2):
    BCELoss = nn.BCEWithLogitsLoss()
    pred2 = torch.argmax(pred2, dim=1) #(bs, 64)
    label = f02img3(pred2)
    loss = BCELoss(pred1, label.cuda())
    return loss

def f02img3(y):
    N = y.size(0)
    length = y.size(1)
    img = torch.zeros([N, 321, length], dtype=torch.float32)
    for i in range(N):
        img[i, y[i], torch.arange(length)] = 1
    return img


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step=0):

    if global_step is not None:
        alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def calculate_contrastive_loss(
    query1, pred1, query2, pred2, temperature=0.5, threshold=0.4
):
    batch_size = query1.shape[0]
    query_num = query1.shape[2]  # (batch_size, 321, query_num)

    # 扩展维度以适配批量计算
    query1 = query1.permute(0, 2, 1)  # (batch_size, query_num, 321)
    query2 = query2.permute(0, 2, 1)  # (batch_size, query_num, 321)

    # 计算正样本相似度 一行是query1对于每一个query2的相似度
    pos_sim = F.cosine_similarity(
        query1.unsqueeze(2), query2.unsqueeze(1), dim=3
    )  # (batch_size, query_num, query_num)

    mask = (
        torch.eye(query_num, device=pos_sim.device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )  # (batch_size, query_num, query_num)
    pos_sim_diag = pos_sim * mask  # (batch_size, query_num, query_num)

    # 计算负样本相似度
    neg_sim = pos_sim.clone() * (1 - mask)  # 反转单位矩阵

    # torch.fill_diagonal_(neg_sim, -float('inf'))  # 确保 i != j 的情况

    # 计算正样本置信度和最大置信度
    positive_confid = pred2.permute(0, 2, 1)  # (batch_size, query_num, 321)
    tch_value, tch_index = torch.max(positive_confid, dim=2)  # (batch_size, query_num)

    # 计算学生置信度
    anchor_confid = pred1.permute(0, 2, 1)  # (batch_size, query_num, 321)
    # 创建索引
    batch_indices = torch.arange(batch_size).unsqueeze(1)  # (batch_size, 1)
    query_indices = torch.arange(query_num).unsqueeze(0)  # (1, query_num)

    # 使用高级索引从 a 中提取值
    stu_value = anchor_confid[
        batch_indices, query_indices, tch_index
    ]  # (batch_size, query_num)

    # 计算有效正样本相似度和负样本相似度
    valid_mask = (stu_value <= tch_value) & (
        tch_value >= threshold
    )  # (batch_size, query_num)
    pos_sim = pos_sim_diag * valid_mask.unsqueeze(2).expand(
        -1, -1, query_num
    )  # 应用掩码

    list2 = []
    for bs in range(batch_size):
        list1 = []
        q = tch_index[bs, :]
        for index in range(query_num):
            v = q[index]
            label = (q != v).int()
            list1.append(label.unsqueeze(0))
        list2.append(torch.cat(list1, dim=0).unsqueeze(dim=0))
    mask2 = torch.cat(list2, dim=0)

    neg_sim = neg_sim * mask2

    # 计算损失
    pos_sim = pos_sim.sum(dim=2)  # (batch_size, query_num)
    neg_sim = neg_sim.sum(dim=2)  # (batch_size, query_num)
    loss = 0
    for bs in range(batch_size):
        x = pos_sim[bs, :]
        y = neg_sim[bs, :]
        labels = (x != 0).int()
        x = x[x != 0]
        y = y * labels
        y = y[y != 0]

        if x.shape != torch.Size([0]) and y.shape != torch.Size([0]):
            loss1 = -torch.log(x + 1e-10 / (x + y + 1e-10))
            loss += loss1.mean()
        else:
            continue

    # loss = -torch.log(pos_sim + 1e-2/ (pos_sim + neg_sim + 1e-2))  # (batch_size, query_num)
    loss = loss / batch_size  # 平均损失

    return loss
if __name__ == '__main__':
    X_train, y_train = load_train_data('train_npy.txt')
    pw = pos_weight(f02img(y_train))
    print(111)