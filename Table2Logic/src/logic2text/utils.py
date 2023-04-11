from collections import defaultdict
import pandas as pd

from intermediate_representation.lf_parser import ASTTree, get_cased_values

def build_table(data):
	table_header = data["table_header"]
	table_cont = data["table_cont"]
	pd_in = defaultdict(list)
	for ind, header in enumerate(table_header):
		for inr, row in enumerate(table_cont):

			# remove last summarization row
			if inr == len(table_cont) - 1 \
					and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or
						 "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
				continue
			pd_in[header].append(row[ind])

	return pd.DataFrame(pd_in)

def build_tree(data, table):
	logic_str = data["logic_str"]

	columns = list(table.columns.values)
	indexes = [str(x) for x in range(1, 20 + 1)]
	cased_values = get_cased_values(logic_str)
	all_table_vals = flatten(preprocess_table(data["table_cont"]))
	cased_values["case1a"] = all_table_vals
	return ASTTree.from_logic_str(logic_str=logic_str, columns=columns, indexes=indexes, cased_values=cased_values)


def preprocess_table(table):
	res = []
	for ind, row in enumerate(table):
		res.append(["row " + str(ind)] + row)
	return res


def linear_table_in(table):
	"""
	get processed linear table for gpt
	"""
	res = ""
	for ind, row in enumerate(table):
		res += (" row " + str(ind) + " : ")
		res += " ; ".join(row)

	return res.strip()


def flatten(t):
	return [item for sublist in t for item in sublist]