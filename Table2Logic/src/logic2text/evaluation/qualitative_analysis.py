from typing import Dict

from intermediate_representation.lf_grammar import Grammar
import json
from intermediate_representation.lf_parser import ASTTree, get_cased_values
from logic2text.utils import build_table, preprocess_table, flatten
from zss import simple_distance
import numpy as np
from logic2text.generation.random_lf_generator import generate_random_lf

def get_sketch(actions):
	sketch = list()
	for ta in actions:
		if type(ta) not in Grammar.terminal_actions():
			sketch.append(ta)
	return sketch

def load_dict(path):
	result = {}
	with open(path) as f:
		dataset = json.load(f)
	for data in dataset:
		logic_str = data["logic_str"]
		pd_table = build_table(data)
		columns = list(pd_table.columns.values)
		# indexes = [str(x) for x in range(1, len(pd_table.index) + 1)]
		indexes = [str(x) for x in range(1, 20 + 1)]
		cased_values = get_cased_values(logic_str)
		all_table_vals = flatten(preprocess_table(data["table_cont"]))
		cased_values["case1a"] = all_table_vals

		tree = ASTTree.from_logic_str(logic_str=logic_str, columns=columns, indexes=indexes, cased_values=cased_values)

		result[data["sha1"]] = tree
	return result

def add_row(div_dict, gold_a, gen_a, div_point, div_percent):
	div_dict["gold_action"].append(None if gold_a is None else str(type(gold_a).__name__))
	div_dict["gold_id_c"].append(None if gold_a is None else gold_a.id_c)
	div_dict["gen_action"].append(None if gen_a is None else str(type(gen_a).__name__))
	div_dict["gen_id_c"].append(None if gen_a is None else gen_a.id_c)
	div_dict["divergence_point"].append(div_point)
	div_dict["divergence_point_percent"].append(div_percent)

def calc_divergence(gold_lf, gen_lf, div_dict):
	gold_actions = gold_lf.to_action_list()
	gen_actions = gen_lf.to_action_list()

	for i, gold_a in enumerate(gold_actions):
		if len(gen_actions) <= i:
			# Gen tree is shorter than gold
			add_row(div_dict, gold_a, None, i+1, (i + 1) / len(gold_actions))
			return
		elif type(gen_actions[i]) != type(gold_a) or gen_actions[i].id_c != gold_a.id_c:
			# Action divergence
			add_row(div_dict, gold_a, gen_actions[i], i + 1, (i + 1) / len(gold_actions))
			return
	# Gen tree is larger
	add_row(div_dict, None, gen_actions[len(gold_actions)], len(gold_actions), 1)

def action_dist(div_dict, dual=False):
	"""
	:param div_dict:
	:param dual: False to use only action type, True to use action + id_c
	:return:
	"""
	dist_dict = {}
	for i, gold_a in enumerate(div_dict["gold_action"]):
		gen_a = div_dict["gen_action"][i]
		gold_id_c = div_dict["gold_id_c"][i]
		gen_id_c = div_dict["gen_id_c"][i]
		key = "{}({})->{}({})".format(gold_a, gold_id_c, gen_a, gen_id_c) if dual else gold_a
		if key not in dist_dict:
			dist_dict[key] = 0
		dist_dict[key] += 1
	# sort
	dist_dict: dict = dict(sorted(dist_dict.items(), key=lambda item: -item[1]))
	# normalize
	dist_dict = add_normalization(dist_dict, len(div_dict["gold_action"]))
	return dist_dict

def add_normalization(dist_dict, total):
	result_dict = {}
	for action, count in dist_dict.items():
		result_dict[action] = (count, count/total)
	return result_dict

def id_c_is_cause(div_dict):
	total = 0
	for i, gold_a in enumerate(div_dict["gold_action"]):
		gen_a = div_dict["gen_action"][i]
		if gold_a == gen_a:
			total += 1
	# sort and return
	return total, total/len(div_dict["gold_action"])

def pintable_distribution(dist_dict):
	result = ", ".join(["{0}: ({1}, {2:.2f})".format(action, dist[0], dist[1]) for action, dist in dist_dict.items()])
	return result

def generate_random_lfs(path):
	result = {}
	with open(path) as f:
		dataset = json.load(f)
	for data in dataset:
		pd_table = build_table(data)
		random_logic_str = generate_random_lf(pd_table)
		columns = list(pd_table.columns.values)
		indexes = [str(x) for x in range(1, 20 + 1)]
		cased_values = get_cased_values(random_logic_str)
		all_table_vals = flatten(preprocess_table(data["table_cont"]))
		cased_values["case1a"] = all_table_vals

		tree = ASTTree.from_logic_str(logic_str=random_logic_str, columns=columns, indexes=indexes, cased_values=cased_values)
		result[data["sha1"]] = tree
	return result

def diff_lf(gen_dataset_path, gold_dataset_path):
	gen_lfs = load_dict(gen_dataset_path)
	gold_lfs = load_dict(gold_dataset_path)
	random_lfs = generate_random_lfs(gold_dataset_path)
	assert len(gen_lfs) == len(gold_lfs)
	divergence_dict = {"sha1": [], "zss_distance": [], "gold_action": [], "gold_id_c": [], "gen_action": [], "gen_id_c": [], "divergence_point": [], "divergence_point_percent": []}
	gold_sizes = []
	gen_sizes = []
	gold_action_distribution = {}
	random_zss_distances = []
	all_zss_distances = []
	for sha1, gold_lf in gold_lfs.items():
		gen_lf = gen_lfs[sha1]
		# Get the LF sizes
		gold_lf_actions = gold_lf.to_action_list()
		gold_sizes.append(len(gold_lf_actions))
		gen_sizes.append(len(gen_lf.to_action_list()))
		# Add action types to global gold action distribution
		for action in gold_lf_actions:
			str_action_type = str(type(action).__name__)
			if str_action_type not in gold_action_distribution:
				gold_action_distribution[str_action_type] = 0
			gold_action_distribution[str_action_type] += 1

		gold_lf_zss = gold_lf.to_zss_tree()
		gen_lf_zss = gen_lf.to_zss_tree()
		# Zhang-Shasha edit distance url: https://pythonhosted.org/zss/
		distance = simple_distance(gold_lf_zss, gen_lf_zss)
		all_zss_distances.append(distance)

		# Get Zhang-Shasha edit distance gold vs random
		random_lf_zss = random_lfs[sha1].to_zss_tree()
		random_zss_distances.append(simple_distance(gold_lf_zss, random_lf_zss))

		# Get first divergent action
		if distance > 0:
			divergence_dict["sha1"].append(sha1)
			divergence_dict["zss_distance"].append(distance)
			calc_divergence(gold_lf, gen_lf, divergence_dict)
	# Average gold tree size <- To see if generated trees tend to be larger
	print("Average gold tree size: {0:.2f}".format(np.mean(gold_sizes)))
	# Average gen tree size <- To see if generated trees tend to be larger
	print("Average gen tree size: {0:.2f}".format(np.mean(gen_sizes)))
	# Average ZSS distance between divergent trees <- To see how different they are, maybe we get a few nodes wrong and the rest is fine
	print("Average ZSS edit distance between divergent trees: {0:.2f}".format(np.mean(divergence_dict["zss_distance"])))
	# Average ZSS distance between divergent trees <- To compare
	print("Average ZSS edit distance between gold and gen vs gold and random trees: {0:.2f} vs {1:.2f}".format(np.mean(all_zss_distances), np.mean(random_zss_distances)))
	# Distribution of gold Action types of the divergence where action type is the cause of divergence <- To see if there's a tendency of the generator to miss in a particular action type
	print("Distribution of gold Action types:")
	# sort
	gold_action_distribution: dict = dict(sorted(gold_action_distribution.items(), key=lambda item: -item[1]))
	gold_action_distribution = add_normalization(gold_action_distribution, np.sum(gold_sizes))
	print(pintable_distribution(gold_action_distribution))
	# Distribution of gold Action types of the divergence where action type is the cause of divergence <- To see if there's a tendency of the generator to miss in a particular action type
	print("Distribution of gold Action types of the divergence where action type is the cause of divergence:")
	print(pintable_distribution(action_dist(divergence_dict)))
	# Distribution of gold-gen action types where action type is the cause of divergence <- To see if there's a tendency to miss generate an action type for another
	print("Distribution of gold-gen action types where action type is the cause of divergence:")
	print(pintable_distribution(action_dist(divergence_dict, dual=True)))
	# Percentage of divergences where action type is correct and id_c is not <- To see if the generator gets the idea of generating actions right but struggles with the pointer
	print("Percentage of divergences where action type is correct and id_c is not:")
	print(id_c_is_cause(divergence_dict)[1])
	# Average divergence point percent <- To see how far in the tree we tend to miss-generate. At the beginning or right at the end. This will be correlated to ZSS distance
	print("Average divergence point percent:")
	print("{0:.2f}".format(np.mean(divergence_dict["divergence_point_percent"])))


if __name__ == '__main__':
	diff_lf("./inferences/exp_3/test.json", "./data/Logic2Text/original_data_fix/test.json")