import json
from tqdm import tqdm
from logic2text.utils import build_table, preprocess_table, flatten

from intermediate_representation.lf_parser import ASTTree, get_cased_values, add_sep_spaces

def execute_all(json_in):
	'''
	execute all logic forms
	'''

	with open(json_in) as f:
		data_in = json.load(f)

	num_all = 0

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
		cased_values["case1a"] = all_table_vals

		# root = ASTNode(logic_str=logic_str)
		# parse_dic(parse(logic_str))
		tree = ASTTree.from_logic_str(logic_str=logic_str, columns=columns, indexes=indexes, cased_values=cased_values)
		gen_str = tree.to_logic_str()
		gen_str = add_sep_spaces(gen_str)
		gen_str += "= true"
		assert logic_str == gen_str
		tree2 = ASTTree.from_logic_str(logic_str=logic_str)
		gen_str2 = tree2.to_logic_str()
		gen_str2 = add_sep_spaces(gen_str2)
		gen_str2 += "= true"
		assert logic_str == gen_str2



if __name__ == '__main__':

	data_path = "../../../data/Logic2Text/original_data_fix/"

	for file_name in ["all_data.json"]:#, "test.json", "train.json", "valid.json"]:
		execute_all(data_path + file_name)
