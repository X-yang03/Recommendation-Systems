# %%
import numpy as np
import heapq
from tqdm import tqdm

data_path = '../data/train.txt'

centerlize = True  # centerlize the rating data or not

train_lines = 0
all_avg = 0
item_rating= {}

with open(data_path, 'r') as f:
    lines = f.readlines()
    user_id = None
    train_lines = len(lines)
    data = {}
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

# %%
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

# %%
# save similarity matrix
np.save('similarity.npy', similarity)

# %%
#similarity = np.load('similarity.npy', allow_pickle=True).item() # 如果保存了中间结果，可以直接读取

# %%
#read test data
test_input = {}
with open('../data/test.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()  # 去除行尾的换行符
        if '|' in line:
            # 这是一个用户的开始
            userid, num_items = line.split('|')  # 分割用户ID和评分项目数量
            test_input[userid] = {'num_items': int(num_items), 'items': []}
        else:
            # 这是一个项目ID
            itemid = line
            test_input[userid]['items'].append(itemid)
lines = None

# %%
def find_top_N_keys(user_id, n, item_id = None, only_keys = True):
    all_similarity = [(i, similarity[i][user_id]) for i in range(user_id)]
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

# %%
#the oridinary version
test_output = {}

for user_id, user_data in (test_input.items()):
    user_id = int(user_id)
    test_output[user_id] = {}
    for item_id in user_data['items']:
        item_id = int(item_id)
        eval = 0
        sim_sum = 0
        N_neighbor = find_top_N_keys(user_id, 10, item_id, only_keys=False)
        for neighbor in N_neighbor:
            if item_id in data[neighbor[0]]['ratings']:
                sim_sum += neighbor[1]
                eval += neighbor[1] * (data[neighbor[0]]['ratings'][item_id])
        if sim_sum != 0:
            eval /= sim_sum
        eval += data[user_id]['sum']/data[user_id]['num_ratings']
        eval = eval if eval <= 100 else 100
        test_output[user_id][item_id] = int(eval) 

print('evaluate done')



# %%
result_path = 'result_normal.txt'
count = 0
with open(result_path, 'w') as f:
    for userid, item_list in test_output.items():
        f.write(f"{userid}|6\n")
        for item, rating in item_list.items():
            f.write(f"{item}  {rating}\n")

print('write done')


