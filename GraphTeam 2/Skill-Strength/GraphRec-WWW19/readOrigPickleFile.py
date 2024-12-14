import pickle
with open('/Users/pujasharma/Desktop/Thesis/Graph-Rec-code/GraphRec-WWW19/data/toy_dataset.pickle', 'rb') as file:
    data = pickle.load(file)
    value = data[1][1] # Access the first element of the tuple

print(value); 
# value = data['key']


