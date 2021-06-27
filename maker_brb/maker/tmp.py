import json

data_path = '/home/liujunwen/MAKER_BRB/data/demodata.json'
with open(data_path, 'r') as f:
    data = json.load(f)
print(data)
