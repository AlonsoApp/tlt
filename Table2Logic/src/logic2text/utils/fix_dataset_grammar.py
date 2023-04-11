import json
from tqdm import tqdm
from logic2text.utils import build_table
from pathlib import Path

from intermediate_representation.lf_parser import ASTTree, get_cased_values, add_sep_spaces


def fix_dataset(original_path, fix_path, file_name):
	'''
	execute all logic forms
	'''

	with open(original_path + file_name) as f:
		data_in = json.load(f)

	num_all = 0
	num_correct = 0

	for data in tqdm(data_in):
		num_all += 1
		# print("Processing: {}".format(num_all))
		logic_str = data["logic_str"]

		pd_table = build_table(data)
		columns = list(pd_table.columns.values)
		indexes = [str(x) for x in range(1, len(pd_table.index) + 1)]
		values = get_cased_values(logic_str)

		tree = ASTTree.from_logic_str(logic_str=logic_str, columns=columns, indexes=indexes, cased_values=values)
		if not tree.is_valid:
			raise Exception("Sample {} has invalid logic".format(num_all - 1))
		fixed_str = tree.to_logic_str()
		fixed_str = add_sep_spaces(fixed_str)
		fixed_str = fixed_str + "= true"
		data["logic_str"] = fixed_str

	with open(fix_path + file_name, 'w', encoding='utf-8') as f:
		json.dump(data_in, f, ensure_ascii=False, indent=4)

	return num_all, num_correct

def run(original_path):
	fix_path = "./data/Logic2Text/original_data_fix_grammar/"
	Path(fix_path).mkdir(parents=True, exist_ok=True)
	for file_name in ["all_data.json", "test.json", "train.json", "valid.json"]:
		fix_dataset(original_path, fix_path, file_name)
	return fix_path


if __name__ == '__main__':
	run("./data/Logic2Text/original_data_fix_ambiguous/")
