import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math

def item_related_item(user_item_dict,user_time_dict):
    related_item = {}
    item_count = defaultdict(int)
    for user, items in tqdm(user_item_dict.items()):
        total_number = 1 + len(items)
        for item_loc1, item in enumerate(items):
            item_count[item] += 1
            related_item.setdefault(item, {})
            for item_loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                position = 0.85**(abs(item_loc1-item_loc2)-1)
                time1 = user_time_dict[user][item_loc1]
                time2 = user_time_dict[user][item_loc2]
                time = 1 - abs(time1 - time2) * 10000
                related_item[item].setdefault(relate_item, 0)
                if item_loc1-item_loc2>0:
                    score = 0.8*1*time*position / math.log(total_number)
                else:
                    score = 1.0*1*time*position / math.log(total_number)
                related_item[item][relate_item] += score

    item_related_item = related_item.copy()

    for i, related_items in tqdm(related_item.items()):
        for j, score in related_items.items():
            item_related_item[i][j] = score / ((item_count[i] * item_count[j]) ** 0.3)

    return item_related_item, user_item_dict



def predict(item_related_item, user_item_dict, user_id):
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]

    for loc, i in enumerate(interacted_items):

        user_list = sorted(item_related_item[i].items(), reverse=True)[0:700]
        for j, score in user_list:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += score * (0.8**loc)

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:700]


# fill user to 50 items
def get_predict(df, pred_col, top_fill):
    top_fill = [int(t) for t in top_fill.split(',')]
    scores = [-1 * i for i in range(1, len(top_fill) + 1)]
    ids = list(df['user_id'].unique())

    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])
    fill_df.sort_values('user_id', inplace=True)
    fill_df['item_id'] = top_fill * len(ids)
    fill_df[pred_col] = scores * len(ids)

    df = df.append(fill_df)
    df.sort_values(pred_col, ascending=False, inplace=True)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)
    df = df[df['rank'] <= 50]
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',', expand=True).reset_index()

    return df

recom_item = []

startphase=7
endphase=9
whole_table = pd.DataFrame()
for c in range(startphase,endphase+1):
    print('phase:', c)
    click_train = pd.read_csv(  '../data/underexpose_train/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])
    click_test = pd.read_csv( '../data/underexpose_test/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c,c),
        header=None,  names=['user_id', 'item_id', 'time'])

    user_col = 'user_id'
    item_col = 'item_id'

    all_click = click_train.append(click_test)
    whole_table = whole_table.append(all_click)
    whole_table = whole_table.drop_duplicates(subset=['user_id','item_id','time'],keep='last')
    whole_table = whole_table.sort_values('time')

    #df = whole_table.copy()
    user_item_ = whole_table.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    user_time_ = whole_table.groupby(user_col)['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

    item_sim_list, user_item = item_related_item(user_item_dict,user_time_dict)

    for i in tqdm(click_test['user_id'].unique()):
        rank_item = predict(item_sim_list, user_item, i)
        for j in rank_item:
            recom_item.append([i, j[0], j[1]])


top50_click = whole_table['item_id'].value_counts().index[:50].values
top50_click = ','.join([str(i) for i in top50_click])

recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
result = get_predict(recom_df, 'sim', top50_click)
result.to_csv('../prediction_result/result.csv', index=False, header=None)
