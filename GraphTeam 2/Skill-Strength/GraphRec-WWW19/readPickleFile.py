# import pandas as pd
# import pickle


# pickleFile = open("/Users/pujasharma/Desktop/Thesis/Graph-Rec-code/GraphRec-WWW19/data/toy_dataset.pickle","rb")
# dataInfo = pickle.load(open("/Users/pujasharma/Desktop/Thesis/Graph-Rec-code/GraphRec-WWW19/data/toy_dataset.pickle", "rb"))
# print (dataInfo)
# df = pd.DataFrame(dataInfo)
# df.to_csv(r'toyDataset.csv')


# # df = pd.read_pickle('/Users/pujasharma/Desktop/Thesis/Graph-Rec-code/GraphRec-WWW19/data/toy_dataset.pickle')




# import libraries
import pickle
import json
import sys
import os

# open pickle file
with open(sys.argv[1], 'rb') as infile:
    obj = pickle.load(infile)

# convert pickle object to json object
json_obj = json.loads(json.dumps(obj, default=str))

# write the json file
with open(
        os.path.splitext(sys.argv[1])[0] + '.json',
        'w',
        encoding='utf-8'
    ) as outfile:
    json.dump(json_obj, outfile, ensure_ascii=False, indent=4)