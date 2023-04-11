import json
from tqdm import tqdm
from pathlib import Path

from logic2text.utils import build_table, flatten
from intermediate_representation.lf_parser import get_cased_values

from logic2text.APIs import *


def fix_dataset(original_path, fix_path, file_name):
	"""
	execute all logic forms
	"""

	with open(original_path + file_name) as f:
		data_in = json.load(f)

	fixed_dataset = []
	removed = 0

	for data in tqdm(data_in):
		logic_str = data["logic_str"]

		pd_table = build_table(data)
		cased_values = get_cased_values(logic_str)
		case1a_values = cased_values["case1a"]

		# compare values against table values
		table_values = flatten(pd_table.values.tolist())
		if do_values_exist_in_table(case1a_values, table_values, fuzzy_eq=False):
			fixed_dataset.append(data)
		else:
			# print("Removed: {}".format(logic_str))
			removed += 1

	with open(fix_path + file_name, 'w', encoding='utf-8') as f:
		print("Saving {} out of {} original values".format(len(fixed_dataset), len(data_in)))
		json.dump(fixed_dataset, f, ensure_ascii=False, indent=4)

	print("Removed: {} samples".format(removed))


def do_values_exist_in_table(values, table_values, fuzzy_eq=False):
	for value in values:
		exists = False
		if fuzzy_eq:
			for t_value in table_values:
				if APIs['eq']['function'](value, t_value) is True:
					exists = True
					break
		else:
			exists = value in table_values
		if not exists:
			return False
	return True


def count_matches(values, table_values, fuzzy_eq=False):
	matches = [0 for _ in values]
	for i, value in enumerate(values):
		for t_value in table_values:
			if fuzzy_eq:
				if t_value == value:
					matches[i] += 1
				elif APIs['eq']['function'](value, t_value):
					matches[i] += 1
	return matches


def run(original_path):
	fix_path = "./data/Logic2Text/original_data_fix_values/"
	Path(fix_path).mkdir(parents=True, exist_ok=True)
	for file_name in ["all_data.json", "test.json", "train.json", "valid.json"]:
		fix_dataset(original_path, fix_path, file_name)
	return fix_path


if __name__ == '__main__':
	run("./data/Logic2Text/original_data_fix_grammar/")
