import json
from tqdm import tqdm
from logic2text.utils import build_table, preprocess_table, flatten
import re

from intermediate_representation.lf_parser import ASTTree, get_cased_values, remove_sep_spaces

def execute_all(json_in):
	'''
	execute all logic forms
	'''

	with open(json_in) as f:
		data_in = json.load(f)

	num_all = 0
	num_correct = 0
	num_error = 0
	errors = []

	for data in tqdm(data_in):
		num_all += 1
		# print("Processing: {}".format(num_all))
		logic = data["logic"]
		logic_str = data["logic_str"]

		pd_table = build_table(data)
		columns = list(pd_table.columns.values)
		#indexes = [str(x) for x in range(1, len(pd_table.index) + 1)]
		indexes = [str(x) for x in range(1, 20 + 1)]
		cased_values = get_cased_values(logic_str)
		all_table_vals = flatten(preprocess_table(data["table_cont"]))
		all_values = all_table_vals + cased_values["case1b"] + cased_values["case2"] + cased_values["case3"]
		cased_values["case1a"] = all_table_vals

		# root = ASTNode(logic_str=logic_str)
		# parse_dic(parse(logic_str))
		tree = ASTTree.from_logic_str(logic_str=logic_str, columns=columns, indexes=indexes, cased_values=cased_values)
		if not tree.is_valid:
			num_error += 1
			errors.append(num_all)
			continue
		#tree.print_graph()
		lf_result = tree.execute(pd_table)
		if lf_result is False:
			tree.print_graph()
			print(num_all)
		action_list = tree.to_action_list()
		tree2 = ASTTree.from_action_list(action_list, columns=columns, indexes=indexes, values=all_values)

		# Compare to logic_str
		logic_str = re.sub(r' ?= ?true', '', logic_str)
		original = remove_sep_spaces(logic_str)
		generated = tree2.to_logic_str()

		if tree.is_valid and tree2.is_valid and original == generated:
			num_correct += 1
		# print("ok")
		else:
			# print("Error on sample: {}".format(num_all))
			# print("Logic: {}".format(logic_str))
			if original != generated:
				print("DIFF {}".format(num_all))
				print(original)
				print(generated)
			num_error += 1
			errors.append(num_all)

	print("All: ", num_all)
	print("Correct: ", num_correct)

	print("Correctness Rate: ", float(num_correct) / num_all)
	print("Errors: {}".format(errors))

	return num_all, num_correct


if __name__ == '__main__':

	data_path = "../../../data/Logic2Text/original_data_fix/"

	for file_name in ["all_data.json"]:#, "test.json", "train.json", "valid.json"]:
		execute_all(data_path + file_name)
