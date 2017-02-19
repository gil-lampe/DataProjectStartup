import json

with open('metadata_mia.json') as json_data:
    d = json.load(json_data)
    print(d)