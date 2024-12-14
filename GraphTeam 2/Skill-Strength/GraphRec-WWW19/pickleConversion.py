import json
import pickle

# Step 1: Read data from JSON file
json_file_path = '/Users/pujasharma/Desktop/Thesis/Graph-Rec-code/GraphRec-WWW19/data/toy_dataset.json'

with open(json_file_path, 'r') as json_file:
    data_from_json = json.load(json_file)

# Step 2: Extract relevant information
history_data = data_from_json['data'][0]

# Assuming the keys inside the first curly braces are dynamic
history_u_lists = {int(key): list(map(int, value)) for key, value in history_data.items()}

# Assuming the keys inside the second curly braces are dynamic
history_ur_lists = {int(key): list(map(int, value)) for key, value in data_from_json['data'][1].items()} 

# Assuming the keys inside the second curly braces are dynamic
history_v_lists = {int(key): list(map(int, value)) for key, value in data_from_json['data'][2].items()}  

# Assuming the keys inside the second curly braces are dynamic
history_vr_lists = {int(key): list(map(int, value)) for key, value in data_from_json['data'][3].items()}     

train_u = list(map(int, data_from_json['data'][4]))
train_v = list(map(int, data_from_json['data'][5]))
train_r = list(map(int, data_from_json['data'][6]))
test_u = list(map(int, data_from_json['data'][7]))
test_v = list(map(int, data_from_json['data'][8]))
test_r = list(map(int, data_from_json['data'][9]))
social_adj_lists = {int(key): set(map(int, value.strip('{}').split(', '))) for key, value in data_from_json['data'][10].items()}
ratings_list = {int(key): int(value) for key, value in data_from_json['data'][11].items()} 

# Save data to a pickle file
pickle_file_path = '/Users/pujasharma/Desktop/Thesis/Graph-Rec-code/GraphRec-WWW19/data/toy1_dataset.pickle'

with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump({
        'history_u_lists': history_u_lists,
        'history_ur_lists': history_ur_lists,
        'history_v_lists': history_v_lists,
        'history_vr_lists': history_vr_lists,
        'train_u': train_u,
        'train_v': train_v,
        'train_r': train_r,
        'test_u': test_u,
        'test_v': test_v,
        'test_r': test_r,
        'social_adj_lists': social_adj_lists,
        'ratings_list': ratings_list,
    }, pickle_file)
