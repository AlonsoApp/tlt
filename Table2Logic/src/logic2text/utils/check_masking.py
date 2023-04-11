import json
import copy
from tqdm import tqdm

from intermediate_representation.lf_grammar import Grammar
from logic2text.utils import build_table, preprocess_table, flatten
import re

from model.example import build_values_extra

from intermediate_representation.lf_parser import ASTTree, get_cased_values, remove_sep_spaces, OOV_TOKEN

def execute_all(json_in, experiment):
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
		logic_str = data["logic_str"]

		table = preprocess_table(data["table_cont"])
		columns = data["table_header"]
		cased_values = get_cased_values(data["logic_str"])

		# we use this to build the pandas datatable later ot execute the LFs
		pd_table = build_table(data)

		table_len = sum([len(row) for row in table])
		column_names = columns
		col_num = len(column_names)
		masked_cases = experiment["masked_cases"].split(";")
		values_extra = build_values_extra(cased_values, experiment["value_cases_in_extra"], masked_cases)

		indexes = [str(x) for x in range(1, 20 + 1)]
		indexes_len = len(indexes)

		# we replace all case 1 values found in gold LF with the entire table values including 'row N' tokens
		cased_values_with_table = cased_values.copy()
		cased_values_with_table["case1a"] = flatten(table)
		# all the values form which the pointer network chooses, including (if set in config) the oov_token
		pointer_values = cased_values_with_table["case1a"] + values_extra  # TODO si case1a esta en masked
		pointer_values.append(OOV_TOKEN)
		pointer_values_len = len(pointer_values)
		ast_tree = ASTTree.from_logic_str(logic_str=logic_str,
											   columns=column_names,
											   indexes=indexes,
											   cased_values=cased_values_with_table,
											   masked_cases=masked_cases)
		tgt_actions = ast_tree.to_action_list()
		ast_tree_from_tgt = ASTTree.from_action_list(tgt_actions, columns, indexes, pointer_values)
		logic_str_tgt = ast_tree_from_tgt.to_logic_str()
		truth_actions = copy.deepcopy(tgt_actions)

		sketch = list()
		if truth_actions:
			for ta in truth_actions:
				if type(ta) not in Grammar.terminal_actions():
					sketch.append(ta)
	print("All: ", num_all)
	print("Correct: ", num_correct)

	print("Correctness Rate: ", float(num_correct) / num_all)
	print("Errors: {}".format(errors))

	return num_all, num_correct


if __name__ == '__main__':

	data_path = "../../../data/Logic2Text/original_data_fix/"

	experiments = [{"value_cases_in_extra": "", "masked_cases": "case1b;case2;case3"},
				   {"value_cases_in_extra": "case1b", "masked_cases": "case2;case3"},
				   {"value_cases_in_extra": "case2", "masked_cases": "case1b;case3"},
				   {"value_cases_in_extra": "case3", "masked_cases": "case1b;case2"},
				   {"value_cases_in_extra": "case1b;case2", "masked_cases": "case3"},
				   {"value_cases_in_extra": "case1b;case3", "masked_cases": "case2"},
				   {"value_cases_in_extra": "case2;case3", "masked_cases": "case1b"}]

	for file_name in ["all_data.json"]:#, "test.json", "train.json", "valid.json"]:
		for experiment in experiments:
			execute_all(data_path + file_name, experiment)
