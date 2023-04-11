import json
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

from logic2text.utils import linear_table_in, build_table

def execute_all(json_in):
	"""
	execute all logic forms
	"""

	with open(json_in) as f:
		data_in = json.load(f)

	topic_lens = []
	header_lens = []
	table_lens = []
	table_row_count = []
	total_lens = []

	for data in tqdm(data_in):
		#print("Processing: {}".format(num_all))
		topic = data["topic"]
		header = " ; ".join(data["table_header"])
		table = linear_table_in(data["table_cont"])

		topic_len = len(topic.split(' '))
		header_len = len(header.split(' '))
		table_len = len(table.split(' '))
		total_len = topic_len + header_len + table_len

		topic_lens.append(topic_len)
		header_lens.append(header_len)
		table_lens.append(table_len)
		table_row_count.append(len(data["table_cont"]))
		total_lens.append(total_len)

	print("Topic")
	print("Mean: {0:.2f}".format(statistics.mean(topic_lens)))
	print("Max: {}".format(max(topic_lens)))
	print("Stdev: {0:.2f}".format(statistics.stdev(topic_lens)))
	print("")
	print("Header")
	print("Mean: {0:.2f}".format(statistics.mean(header_lens)))
	print("Max: {}".format(max(header_lens)))
	print("Stdev: {0:.2f}".format(statistics.stdev(header_lens)))
	print("")
	print("Table")
	print("Mean: {0:.2f}".format(statistics.mean(table_lens)))
	print("Max: {}".format(max(table_lens)))
	print("Stdev: {0:.2f}".format(statistics.stdev(table_lens)))
	print("")
	print("Table rows")
	print("Mean: {0:.2f}".format(statistics.mean(table_row_count)))
	print("Max: {}".format(max(table_row_count)))
	print("Stdev: {0:.2f}".format(statistics.stdev(table_row_count)))
	print("")
	print("Total")
	print("Mean: {0:.2f}".format(statistics.mean(total_lens)))
	print("Max: {}".format(max(total_lens)))
	print("Stdev: {0:.2f}".format(statistics.stdev(topic_lens)))
	print("")

	bigger = [x for x in total_lens if x>512]
	print(">512: {}/{} {:.2f}%".format(len(bigger), len(total_lens), (len(bigger)/ len(total_lens))*100))

	sns.histplot(data=table_row_count)
	plt.show()



if __name__=='__main__':


	data_path = "../../../data/Logic2Text/original_data_fix/"

	for file_name in ["all_data.json"]:#, "test.json", "train.json", "valid.json"]:
		execute_all(data_path+file_name)

