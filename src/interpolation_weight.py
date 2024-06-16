import numpy as np
from tqdm import tqdm
import os
import heapq

similarity = None
data = None
test_input = None
all_avg = None
item_rating = None

bias = {}
item_bias = {}

def read_train_input(data_path='../data/train.txt', centerlize=False):
    train_lines = 0
    all_avg = 0
    item_rating= {}
    data = {}
    with open(data_path, 'r') as f:
        lines = f.readlines()
        user_id = None
        train_lines = len(lines)
        for line in lines:
            line = line.strip()
            if '|' in line:  # user line
                train_lines -= 1
                if(user_id != None and centerlize == True):
                    avg = data[user_id]['sum'] / data[user_id]['num_ratings']
                    data[user_id]['ratings'].update({k: v - avg for k, v in data[user_id]['ratings'].items()})
                    data[user_id]['norm'] = sum(x**2 for x in data[user_id]['ratings'].values())**0.5

                user_id, num_ratings = line.split('|')
                user_id = int(user_id)
                data[user_id] = {}
                data[user_id]['num_ratings'] = int(num_ratings)
                data[user_id]['ratings'] = {}
                data[user_id]['sum'] = 0
            else:  # rating line
                item_id, score = map(int, line.split())
                data[user_id]['ratings'][item_id] = score
                data[user_id]['sum'] += score
                all_avg += score
                if item_id not in item_rating:
                    item_rating[item_id] = {'num': 0, 'sum': 0}
                item_rating[item_id]['num'] += 1
                item_rating[item_id]['sum'] += score
        if centerlize == True:
            avg = data[user_id]['sum'] / data[user_id]['num_ratings']
            data[user_id]['ratings'].update({k: v - avg for k, v in data[user_id]['ratings'].items()})
            data[user_id]['norm'] = sum(x**2 for x in data[user_id]['ratings'].values())**0.5

    all_avg /= train_lines
    lines = None
    return data, item_rating, all_avg

def read_test_data(path = '../data/test.txt'):
    test_input = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 去除行尾的换行符
            if '|' in line:
                # 这是一个用户的开始
                userid, num_items = line.split('|')  # 分割用户ID和评分项目数量
                userid = int(userid)
                test_input[userid] = {'num_items': int(num_items), 'items': []}
            else:
                # 这是一个项目ID
                itemid = int(line)
                test_input[userid]['items'].append(itemid)
    lines = None
    return test_input

def write_result(result_path, test_output):
    with open(result_path, 'w') as f:
        for userid, item_list in test_output.items():
            f.write(f"{userid}|6\n")
            for item, rating in item_list.items():
                f.write(f"{item}  {rating}\n")

    print('write done')

def cal_similarity():
    similarity = {}
    for i, (userid1, user1_data) in tqdm(enumerate(data.items()), total=len(data)):
        similarity[userid1] = {}
        for j, (userid2, user2_data) in enumerate(data.items()):
            if i >= j:
                continue
            if  user1_data['norm'] == 0 or  user2_data['norm'] == 0:
                similarity[userid1][userid2] = 0
                continue
            else:
                cos_sim = 0.0
                for item, rating in user1_data['ratings'].items():
                    if item in user2_data['ratings']:
                        cos_sim += rating * user2_data['ratings'][item]
                cos_sim = cos_sim / (user1_data['norm'] * user2_data['norm'])
                similarity[userid1][userid2] = cos_sim
    print("Similarity matrix calculated")
    return similarity

def find_top_N_keys(user_id, n, item_id = None, only_keys = True):
    all_similarity = [(i,similarity[i][user_id]) for i in range(user_id)]
    all_similarity.extend(list(similarity[user_id].items()))
    if item_id == None: #不要求item_id在dictionary中
        top_n_items = heapq.nlargest(n, all_similarity, key=lambda x: x[1])
    else:   
        top_n_items = heapq.nlargest(n, ((k, v) for (k, v) in all_similarity if item_id in data[k]['ratings']), key=lambda item: item[1])
    if only_keys:
        top_n_keys = [key for (key, value) in top_n_items ]
        return top_n_keys
    else:
        return top_n_items

def get_interpolation_weight(userid, itemid, N = 10):
    initial_lr = 1e-2
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = 0     

    epsilon = 10 #收敛条件，RMSE小于10停止
    N_neighbor = find_top_N_keys(userid, N, item_id = itemid,only_keys=False) #为了加速计算，只选择相似度最高的N个邻居
    N = len(N_neighbor) #实际邻居数量

    sim_sum = np.sum([sim for (id,sim) in N_neighbor])
    if(sim_sum == 0):
        return None
    w = np.array([sim/sim_sum for (id,sim) in N_neighbor])  #初始化权重, 并归一化
    
    mse_eps = 1
    mse_old = 0

    m = np.zeros_like(w)
    v = np.zeros_like(w) #Adam参数

    rating_matrix = np.zeros((N, len(data[userid]['ratings']))) #评分矩阵
    true_rating = np.array(list(data[userid]['ratings'].values())) #真实评分，作为标签
    user_bias = np.zeros(len(data[userid]['ratings']))  #用户偏差

    grad = np.zeros(N)
    #初始化评分矩阵
    for i,itemid in enumerate(data[userid]['ratings'].keys()): #因为计算时有w *（r_yi - b_yi），所以这里先将(r_yi - b_yi)算出来
        user_bias[i] = bias[userid] + item_bias[itemid] + all_avg
        for j,(neighbor,sim_) in enumerate(N_neighbor):
            if itemid in data[neighbor]['ratings']: 
                rating_matrix[j][i] = data[neighbor]['ratings'][itemid]  - (bias[neighbor] + item_bias[itemid] + all_avg)
            else:
                rating_matrix[j][i] = 0
    #迭代训练
    while True:
        t += 1 #迭代次数，用于Adam更新
        eval = np.dot(w, rating_matrix)  #使用矩阵乘法形式，加速计算
        nonzero_indices = np.nonzero(eval)
        eval[nonzero_indices] += user_bias[nonzero_indices]  #加上用户偏差，得到最终预测评分   
         
        loss = np.sum((eval[nonzero_indices] - true_rating[nonzero_indices])**2) #计算损失
        grad = np.array([(2*(eval - true_rating) * rating_matrix[N_neighbor.index(n)]).sum() for n in N_neighbor])
        MSE = loss/len(nonzero_indices[0])

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        diff = abs(MSE - mse_old)
        if(diff < mse_eps):
            break
        mse_old = MSE
        w = w - initial_lr * m_hat / (np.sqrt(v_hat) + eps)
        if(MSE < epsilon**2):
            break
    W = {k: v for (k,k1), v in zip(N_neighbor, w)}
    return W

def eval_result():
    print('evaluating... ')
    test_output = {}

    for user_id, user_data in (tqdm(test_input.items())):
        user_id = int(user_id)
        test_output[user_id] = {}
        for item_id in user_data['items']:
            item_id = int(item_id)
            eval = 0
            if item_id not in item_rating: #物品没有评分
                test_output[user_id][item_id] = int(data[user_id]['sum']/data[user_id]['num_ratings']) #直接用用户的平均评分
                continue
            W = get_interpolation_weight(user_id, item_id, 10) #插值权重
            if W == None: #无法插值
                test_output[user_id][item_id] = data[user_id]['sum']/data[user_id]['num_ratings']
                continue
            for neighbor in W.keys():
                eval += W[neighbor] * (data[neighbor]['ratings'][item_id] - (bias[neighbor] + item_bias[item_id] + all_avg))
            
            baseline =bias[user_id] + item_bias[item_id] + all_avg
            eval += baseline
            eval = min(100, eval)
            eval = max(0, eval)
            test_output[user_id][item_id] = int(eval)
                

    print('evaluate done')
    return test_output

train_data_path='../data/train.txt' #数据路径，和src在同一目录下
test_output_path = 'result_weighted.txt' #输出路径
similarity_path = '../similarity.npy' #相似度矩阵路径

if not os.path.exists(train_data_path):
    print(f"File '{train_data_path}' does not exist. Please check the path and rerun the script.")
    exit(1)
test_input = read_test_data()

if os.path.exists(similarity_path):
    similarity = np.load(similarity_path, allow_pickle=True).item()
    data, item_rating, all_avg = read_train_input(train_data_path, centerlize=False)
else:
    data, item_rating, all_avg = read_train_input(train_data_path, centerlize=True)
    print(f"File '{similarity_path}' does not exist. Calculating similarity matrix... Takes 1-2 hours.")
    similarity = cal_similarity(data)
    np.save(similarity_path, similarity)
    data, item_rating, all_avg = read_train_input(train_data_path, centerlize=False)

print('get similarity matrix.')
for userid in data.keys():
        bias[userid] = data[userid]['sum']/data[userid]['num_ratings'] - all_avg

for itemid in item_rating.keys():
    item_bias[itemid] = item_rating[itemid]['sum']/item_rating[itemid]['num'] - all_avg

result = eval_result()
write_result(test_output_path, result)