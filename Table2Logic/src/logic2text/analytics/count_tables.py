import json

import numpy as np
from tqdm import tqdm

def execute_all(json_in):
	'''
	execute all logic forms
	'''

	with open(json_in) as f:
		data_in = json.load(f)

	tables = {}

	for data in tqdm(data_in):
		# print("Processing: {}".format(num_all))
		url = data["url"]
		if url not in tables:
			tables[url] = 0
		tables[url] += 1

	values = []
	for key in tables.keys():
		values.append(tables[key])
	print("Tables")
	print("Unique: {0}".format(len(tables.keys())))
	print("Mean: {0:.2f}".format(np.mean(values)))
	print("max: {0:.2f}".format(np.max(values)))


if __name__ == '__main__':

	data_path = "../../../data/Logic2Text/original_data_fix_tokens/"

	for file_name in ["all_data.json"]:#, "test.json", "train.json", "valid.json"]:
		execute_all(data_path + file_name)
