import os
import json
from transformers import BertTokenizer

from logic2text.utils import preprocess_table, build_table
from intermediate_representation.lf_parser import get_cased_values

# Make this get the params form config
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_data_new(dataset_path, use_small=False):
	with open(dataset_path) as f:
		dataset = json.load(f)
	data = []
	for sample in dataset:
		topic = sample["topic"]
		topic = split_into_tokens(topic)
		table = preprocess_table(sample["table_cont"])
		columns = sample["table_header"]
		cased_values = get_cased_values(sample["logic_str"])

		# we use this to build the pandas datatable later ot execute the LFs
		pd_table = build_table(sample)
		logic_str = sample["logic_str"]
		sent = sample["sent"]

		data_sample = {"topic": topic, "table": table, "columns": columns, "cased_values": cased_values,
					   "logic_str": logic_str, "sent": sent, "pd_table": pd_table}
		data.append(data_sample)
	return data


def load_dataset(dataset_dir, dataset_file, use_small=False):
	dataset_path = os.path.join(dataset_dir, dataset_file)
	return load_data_new(dataset_path, use_small)

def split_into_tokens(text):

	all_sub_token = tokenizer.tokenize(text)

	no_subword_tokens = []

	for sub_token in all_sub_token:
		if len(sub_token) > 2 and sub_token[0:2] == '##':
			no_subword_tokens[-1] += sub_token[2:]
		else:
			no_subword_tokens.append(sub_token)
	return no_subword_tokens

if __name__ == '__main__':
	load_dataset("../../data/Logic2Text/original_data_fix")
