import re
from typing import List, Dict

import pandas as pd

from execution.APIs import APIs, fuzzy_match_eq
from gpt_base.table2logic.lf_grammar import *
from treelib import Tree as TreePlt

OOV_TOKEN = "[OOV]"

class ASTNode(object):
	def __init__(self, body: str, args: [], action: Action = None, value_case: str = None, masked: bool = False):
		"""
		:param body:
		:param args:
		:param action:
		:param value_case: 0 no value, 1 value from table, 2 value result of arithmetic op, 3 value chosen by LF author
		"""
		self.body = body
		self.action = action
		self.args = args
		self.value_case = value_case
		self.masked = masked

	@classmethod
	def from_logic_str(cls, func: str, args: []):
		if len(args) == 0:
			return ASTNode(func, [])
		action = cls._action(func)
		node_args: List[ASTNode] = []
		for arg in args:
			if type(arg) is dict:
				arg_fn = next(iter(arg))
				node_args.append(ASTNode.from_logic_str(arg_fn, arg[arg_fn]))
			else:
				node_args.append(ASTNode.from_logic_str(arg, []))

		return ASTNode(func, node_args, action)

	@classmethod
	def from_action_list(cls, actions: List[Action], predefined_values=None, padding=False):
		actions = actions.copy()
		return ASTNode._req_from_action_list(actions, predefined_values, padding)

	@staticmethod
	def _req_from_action_list(actions: List[Action], predefined_values=None, padding=False):
		"""
		if padding = True we expect only sketch actions and we add the corresponding terminal action based on the
		expected next action. We don't propagate the recursion further because it is a terminal action
		:param actions:
		:param predefined_values:
		:param padding:
		:return:
		"""
		action = actions.pop(0)
		next_actions = action.get_next_action()
		args = []
		for next_action in next_actions:
			if padding and next_action in Grammar.terminal_actions():
				args.append(ASTNode(body='', args=[], action=next_action(0)))
			else:
				node = ASTNode._req_from_action_list(actions, predefined_values, padding)
				args.append(node)
		# body
		body = ''
		masked = False
		if type(action) in Grammar.terminal_actions():
			if predefined_values is not None:
				body = predefined_values[type(action)][action.id_c]
				if body == OOV_TOKEN:
					masked = True
		else:
			for x in action.production.split(' ')[1:]:
				if x in keywords:
					body = x
					break
		return ASTNode(body, args, action, masked=masked)

	def set_action(self, action):
		self.action = action

	def set_args(self, args: []):
		self.args = args

	def validate(self):
		"""
		When parsing from logic_str
		Validates and defines undefined actions of leaf nodes. And expands the nodes of Obj that consist only on N or V
		:return:
		"""
		result = True
		next_actions = self.action.get_next_action()
		for i, (arg, next_action) in enumerate(zip(self.args, next_actions)):
			if arg.action is None and arg.body == "all_rows" and type(next_action) is type(View):
				# This is an edge case were we have skipped the action assignation on init to protect against using
				# keywords as column, or value names
				arg.set_action(View(View.id_all_rows))
			elif next_action is Obj and type(arg.action) != Obj:
				# Obj(4) gets direct V or N in str parsing but in graph we need an extra node Obj->V
				if arg.action is None:
					v_node = ASTNode(arg.body, args=[], action=V(0))
					obj_node = ASTNode('', args=[v_node], action=Obj(Obj.id_V))  # V
					self.args[i] = obj_node
				elif type(arg.action) == N:
					obj_node = ASTNode('', args=[arg], action=Obj(Obj.id_N))  # N
					self.args[i] = obj_node
					result = result and self.args[i].validate()
				else:
					print("ERROR: Tree action: Obj should be N or V not {}".format(arg.action))
					return False
			elif arg.action is None and next_action in Grammar.terminal_actions():
				arg.set_action(next_action(0))
			elif type(arg.action) == next_action:
				result = arg.validate()
				if not result:
					return False
			elif (self.body == 'str_hop' or self.body == 'num_hop') and type(arg.action) == View:
				# The original LF grammar presented in Logic2Text has an inconsistency where
				# hop (which returns a unique value of a column given a row) gets a View instead of a Row. This is
				# grammatically incorrect but is present in the 25% of that dataset samples. To fix this we change the
				# 'hop' keyword of all 'hop View C' rules to 'hop_first'. Making it coherent through all the grammar.
				#print("Changed hop to hop_first")
				self.body = 'str_hop_first' if self.body == 'str_hop' else 'num_hop_first'
				self.action = Obj(Obj.id_str_hop_first)  if self.body == 'str_hop_first' else Obj(Obj.id_num_hop_first)
				result = arg.validate()
				if not result:
					return False
			else:
				print("ERROR: {} expected {} but got {}".format(self.body, next_action.__name__, arg.action))
				return False
		return result

	def get_values(self, values: List[str]):
		if type(self.action) == V:
			values.append(self.body)
		for arg in self.args:
			arg.get_values(values)

	def get_cased_values(self, values: Dict):
		"""
		Fills the given values Dict with all the values classified in case1a, case1b, case2 and case3 depending on the graph structure
		ATTENTION: duplicate values are not returned size(cased_values) != size(V_nodes_in_tree)
		:param values:
		:param cntx:
		:return:
		"""
		if type(self.action) == V and self.body not in values[self.value_case]:
			values[self.value_case].append(self.body)
		for arg in self.args:
			arg.get_cased_values(values)

	def mask_values(self, masked_cases: List[str]):
		"""
		Sets mask as True for every value whose case is in masked_cases list
		:param masked_cases:
		:return:
		"""
		if type(self.action) == V and self.value_case in masked_cases:
			self.masked = True
		for arg in self.args:
			arg.mask_values(masked_cases)

	def assign_id_to_terminal_nodes(self, predefined_values, masked_cases):
		"""
		Given a list of columns, indexes, values_case1, values_case2 and values_case3 assigns the index of the matching
		value of each list to its corresponding node type. Sets id_c of C, I and V
		:param predefined_values: {C: [], I: [], V: {"case1a":[], "case1b":[], "case2":[], "case3":[]}} case 1 must contain all table values including 'row N' tokens
		:param masked_cases: we need to know which vale cases will be masked to calculate the index offset
		:return:
		"""
		if type(self.action) in Grammar.terminal_actions():
			self.action.id_c = self._find_id_c(predefined_values, masked_cases)
		for arg in self.args:
			arg.assign_id_to_terminal_nodes(predefined_values, masked_cases)

	def assign_case_to_values(self, cntxt=None):
		"""
		Considering the graph structure in which V values are, assigns its corresponding case (1a,1b,2,3)
		:param values:
		:param cntxt:
		:return:
		"""
		# if we are in a pre-value Object
		if type(self.action) == Obj and self.action.id_c == Obj.id_V:
			found = False
			if cntxt.action.potential_case_value() == 3:
				# the parent action is considered a potential case 3 action "less, greater than..."
				self.args[0].value_case = "case3"
				found = True
			if cntxt.action.potential_case_value() == 4:
				# the parent action is considered a potential case 1b action "filter"
				self.args[0].value_case = "case1b"  # set to "case1a" to merge 1a + 1b
				#self.args[0].case1_col = find_col_in_node(self)
				found = True
			elif type(cntxt.action) == Stat and cntxt.action.potential_case_value() == -1:
				# the parent action is eq and we have to check its siblings to see if case 1 or case 2
				siblings: List = cntxt.args.copy()
				siblings.remove(self)
				for arg in siblings:
					if arg.action.potential_case_value() == 1 or \
							(arg.action.potential_case_value() == -1 and arg.args[0].action.potential_case_value() == 1):
						# parent is eq and sibling is case1a action (like hop). Thus, this value is case1a
						# or this is an Obj N and its N child is case 1 (like max)
						self.args[0].value_case = "case1a"
						# let's save the col this value comes from to scope its id later on
						# self.args[0].case1_col = find_col_in_node(arg)
						found = True
						break
					if arg.action.potential_case_value() == 2 or \
							(arg.action.potential_case_value() == -1 and arg.args[0].action.potential_case_value() == 2):
						# parent is eq and sibling is case2 action (like diff). Thus, this value is case2
						# or this is an Obj N and its N child is case 2
						self.args[0].value_case = "case2"
						found = True
						break
			if not found:
				# none of the above conditions apply, it must be a case3
				self.args[0].value_case = "case3"
				#print("---------------Rare case found---------------")
				#tree = TreePlt()
				#cntxt.req_print_graph(tree)
				#tree.show()
				# raise Exception("Value case not found in: {parent}: ")

		for arg in self.args:
			arg.assign_case_to_values(cntxt=self)

	def _find_id_c(self, predefined_values, masked_cases=None, fuzzy_eq=False):
		"""
		Gets the id_c of the C, I or V action for a text that matches a value of its corresponding predefined_values list
		:param text: the text for which we want to get the id_c
		:param action: the action of the current node
		:param predefined_values: {C: [], I: [], V: {"case1a":[], "case1b":[], "case2":[], "case3":[]}} case 1 must contain all table values including 'row N' tokens
		:param value_case: in case this is a V node, which value case is this (None = no case, 'case1a', 'case1b', 'case2' , 'case3')
		:param fuzzy_eq: in case we want to merge cases 1a and 1b, we need fuzzy_eq if we want to find the id for some 1b values
		:return:
		"""
		masked_cases = [] if masked_cases is None else masked_cases
		if type(self.action) is V:
			if self.masked is True:
				# if masked we assign the last index +1. Index that would correspond to the oov_token
				return sum([len(val) if key not in masked_cases else 0 for key, val in predefined_values[V].items()])
			if self.value_case == 'case1a':
				# First we try finding the case1 value with == which is stricter
				for i, val in enumerate(predefined_values[V]["case1a"]):
					if self.body == val:
						return i
				if fuzzy_eq:
					# If == doesn't find any matches we go to fuzz eq
					for i, val in enumerate(predefined_values[V]["case1a"]):
						if APIs['eq']['function'](self.body, val) is True or APIs['str_eq']['function'](self.body, val) is True:
							return i
					# eq and str_eq functionality is not the same as the one used in fuzzy_match filters of Logic2Text APIs
					for i, val in enumerate(predefined_values[V]["case1a"]):
						if fuzzy_match_eq(val, self.body):
							return i

			elif self.value_case == 'case1b':
				i = predefined_values[V]["case1b"].index(self.body)
				offset = 0 if "case1a" in masked_cases else len(predefined_values[V]["case1a"])
				return offset + i
			elif self.value_case == 'case2':
				i = predefined_values[V]["case2"].index(self.body)
				offset = 0 if "case1a" in masked_cases else len(predefined_values[V]["case1a"])
				offset += 0 if "case1b" in masked_cases else len(predefined_values[V]["case1b"])
				return offset + i
			elif self.value_case == 'case3':
				i = predefined_values[V]["case3"].index(self.body)
				offset = 0 if "case1a" in masked_cases else len(predefined_values[V]["case1a"])
				offset += 0 if "case1b" in masked_cases else len(predefined_values[V]["case1b"])
				offset += 0 if "case2" in masked_cases else len(predefined_values[V]["case2"])
				return offset + i
		else:
			return predefined_values[type(self.action)].index(self.body)


	@staticmethod
	def _action(func: str):
		"""
		A function in Logic2Text is a keyword of our AST grammar. Each keyword is only related to only one prod rule
		Therefore, if func == keyword of the prod rule, we know the specific prod rule (and action) this part of the LF
		refers to.
		BEWARE: We are lucky our grammar only has one unique keyword for each production rule, therefore, if we know the
		keyword we know the production rule. If we update our grammar and the same keyword can be found in different
		production rules we have to check if the args comply with the production rule's grammar
		:param func:
		:return:
		"""
		g = Grammar()
		# for every action class in the grammar
		for action_cls in g.sketch_actions():
			productions = g.get_production(action_cls)
			# for every production rule in this action class
			for prod_id, production in enumerate(productions):
				# for each token in the production rule
				for token in production.split(' ')[1:]:
					if token in keywords and token == func:
						return action_cls(prod_id)
		# raise NotImplementedError("The function: {} can't be found in the grammar".format(func))
		return None

	def to_logic_str(self):
		if len(self.args) == 0:
			return self.body

		str_args = [x.to_logic_str() for x in self.args]
		if self.body == "":
			# bridge node e.g Obj->N or Obj->V jump right to print args
			result = ";".join(str_args)
		else:
			result = "{func}{{{args}}}".format(func=self.body, args=";".join(str_args))
		return result

	def to_action_list(self):
		result = [self.action]
		for arg in self.args:
			result += arg.to_action_list()
		return result

	def execute(self, table: pd.DataFrame):
		if len(self.args) == 0:
			return table if self.body == "all_rows" else self.body

		args = [x.execute(table) for x in self.args]
		if self.body == "":
			# bridge node e.g Obj->N or Obj->V jump right to print args
			result = args[0]
		else:
			result = APIs[self.body]["function"](*args)
			if APIs[self.body]["output"] == "str":
				result = str(result)
		return result

	def req_print_graph(self, tree, parent=None):
		"""
		Recursive function to build a TreePlt to print the graph in a readable way
		:param tree:
		:param parent:
		:return:
		"""
		if parent is None:
			node = tree.create_node(self.body)
		else:
			if self.body == "":
				# bridge node e.g Obj->N or Obj->V jump right to print args
				node = parent
			else:
				node = tree.create_node(self.body, parent=parent)

		for arg in self.args:
			if len(arg.args) == 0:
				text = arg.body if arg.masked is False else "{} (masked)".format(arg.body)
				tree.create_node(text, parent=node)
			else:
				arg.req_print_graph(tree, parent=node)


class ASTTree(object):
	def __init__(self, root: ASTNode, is_valid: bool = False):
		self.root = root
		self.is_valid = is_valid

	@classmethod
	def from_logic_str(cls, logic_str: str, columns: List[str] = None, indexes: List[str] = None,
					   cased_values: Dict[str, List[str]] = None, masked_cases: List[str] = None):
		"""
		:param logic_str: str representation of the logic form
		:param columns: list of columns
		:param indexes: list of all indexes []
		:param cased_values: {"case1a":[], "case1b":[], "case2":[], "case3":[]} case 1 must contain all table values including 'row N' tokens
		:param masked_cases: ["case2"] all cases in this list will be marked as masked and its id_c will be the index of oov_token (len+1)
		:return:
		"""
		# remove the last '=true' part
		logic_str = re.sub(r' ?= ?true', '', logic_str)
		logic = _parse_dic(_parse(logic_str)[0])[0]
		first_fn = next(iter(logic))
		args = logic[first_fn]
		root = ASTNode.from_logic_str(first_fn, args)
		predefined_values = None
		if columns is not None and indexes is not None and cased_values is not None:
			predefined_values = {C: columns, I: indexes, V: cased_values}
		is_valid = root.validate()
		root.assign_case_to_values()
		if masked_cases is not None and masked_cases != ['']:
			root.mask_values(masked_cases)
		if predefined_values is not None:
			root.assign_id_to_terminal_nodes(predefined_values, masked_cases)
		return ASTTree(root, is_valid)

	@classmethod
	def from_action_list(cls, actions: List[Action], columns: List[str]=None, indexes: List[str]=None, values: List[str]=None, padding=False):
		"""
		:param actions:
		:param columns:
		:param indexes:
		:param values: list of all values, containing "row n"+case1a+case1b+case2+case3+oov_token. The same list we feed the pointer
		:return:
		"""
		predefined_values = None
		if columns is not None and indexes is not None and values is not None:
			predefined_values = {C: columns, I: indexes, V: values}
		root = ASTNode.from_action_list(actions, predefined_values, padding=padding)
		is_valid = root.validate()
		root.assign_case_to_values()
		return ASTTree(root, is_valid)

	def to_logic_str(self):
		return self.root.to_logic_str()

	def to_action_list(self):
		return self.root.to_action_list()

	def execute(self, table: pd.DataFrame):
		try:
			return self.root.execute(table)
		except:
			#print(traceback.format_exc())
			return False

	def print_graph(self):
		tree = TreePlt()
		self.root.req_print_graph(tree)
		print()
		tree.show()

	def get_sketch(self):
		sketch = list()
		actions = self.to_action_list()
		if actions:
			for ta in actions:
				if type(ta) not in Grammar.terminal_actions():
					sketch.append(ta)
		return sketch


# Helpers
def _append(lst: [], idx_lst: [int], obj):
	"""
	Given a nested list, a list of indexes indicating the position of the appended object and object. Appends the
	object to the last element of the index. This works in the same way we could append an object to a tensor
	:param lst: nested list in which the object will be appended
	:param idx_lst: [0, 1, 2] a list of indexes indicating the possition of the appended object. In this case obj will
	be appended to the element in position 2 of the list in the position 1 in the list of position 0 of the main list
	:param obj: object to append, can be str or list [[]]
	:return:
	"""
	curr_lst = lst
	for i, idx in enumerate(idx_lst):
		if i + 1 < len(idx_lst):
			curr_lst = curr_lst[idx]
		else:
			curr_lst[idx] += obj


def remove_sep_spaces(s: str) -> str:
	# clean spaces between separators
	for sep in ['{', '}', ';']:
		s = re.sub(r' ?{} ?'.format(sep), sep, s)
	return s


def add_sep_spaces(s: str) -> str:
	# add spaces between separators
	for sep in ['{', '}', ';']:
		s = re.sub(r'(?<! ){}'.format(sep), ' ' + sep, s)
		s = re.sub(r'{}(?! )'.format(sep), sep + ' ', s)
	return s


def _parse(s: str) -> []:
	s = remove_sep_spaces(s)
	idx: List[int] = [0, 0]
	queue = [[""]]
	for char in s:
		if char == "{":
			_append(queue, idx[:-1], [[""]])
			idx[-1] += 1
			idx.append(0)
		elif char == ";":
			_append(queue, idx[:-1], [""])
			idx[-1] += 1
		elif char == "}":
			idx.pop(-1)
		else:
			_append(queue, idx, char)
	return queue


def _parse_dic(lst: []):
	"""
	Gets hierarchy well. Converts form:
	[['eq', ['hop', ['argmax', ['all_rows', 'average'], 'player'], 'alec bedser']]]
	to
	[{'eq': [{'hop': [{'argmax': ['all_rows', 'average']}, 'player']}, 'alec bedser']}]
	:param lst: lst from parse
	:return:
	"""
	result_lst = []
	while len(lst) > 0:
		if len(lst) == 1:
			result_lst.append(lst[0])
			lst = []
		elif type(lst[0]) is str and type(lst[1]) is list:
			result_lst.append({lst[0]: _parse_dic(lst[1])})
			lst = lst[2:]
		elif type(lst[0]) is str and type(lst[1]) is str:
			result_lst.append(lst[0])
			lst = lst[1:]
		else:
			raise Exception()

	return result_lst


def get_values(logic_str):
	tree = ASTTree.from_logic_str(logic_str)
	values = []
	tree.root.get_values(values)
	return values

def get_cased_values(logic_str):
	tree = ASTTree.from_logic_str(logic_str=logic_str)
	cased_values = {'case1a': [], 'case1b': [], 'case2': [], 'case3': []}
	tree.root.get_cased_values(cased_values)
	return cased_values

def find_col_in_node(node:ASTNode):
	"""
	Given an ASTNode that may have a C as a child. returns the column name of that C child
	:param node:
	:return:
	"""
	for arg in node.args:
		if type(arg.action) is C:
			return arg.body
	return None

# Tests
def test_cased_values():
	# logic_str = "eq { hop { nth_argmax { all_rows ; attendance ; 3 } ; competition } ; danish superliga 2005 - 06 } = true"
	logic_str = "eq { diff { hop_first { filter_eq { all_rows ; incumbent ; bruce vento } ; opponent } ; hop_first { filter_eq { all_rows ; incumbent ; gerry sikorski } ; opponent } } ; .10 % } = true"
	tree = ASTTree.from_logic_str(logic_str=logic_str)
	oot_values = {1: [], 2: [], 3: []}
	tree.root.get_cased_values(oot_values)
	values = []
	tree.root.get_values(values)
	assert len(oot_values.values()) == len(values)

def test_cased_values_02():
	# logic_str = "eq { hop { nth_argmax { all_rows ; attendance ; 3 } ; competition } ; danish superliga 2005 - 06 } = true"
	logic_str = "str_eq { str_hop { nth_argmin { all_rows ; population ; 2 } ; english name } ; liancheng county } = true"
	cased_values = get_cased_values(logic_str)
	print(cased_values)

def test_to_str():
	# logic_str = "eq { hop { nth_argmax { all_rows ; attendance ; 3 } ; competition } ; danish superliga 2005 - 06 } = true"
	logic_str = "and { greater { num_hop_first { filter_str_eq { all_rows ; year ; 1995 } ; pts } ; num_hop_first { filter_str_eq { all_rows ; year ; 1991 } ; pts } } ; and { eq { num_hop_first { filter_str_eq { all_rows ; year ; 1995 } ; pts } ; 13 } ; eq { num_hop_first { filter_str_eq { all_rows ; year ; 1991 } ; pts } ; 1 } } } = true"
	#logic_str = "and { only { filter_str_eq { all_rows ; largest ethnic group ( 2002 ) ; slovaks } } ; str_eq { str_hop_first { filter_str_eq { all_rows ; largest ethnic group ( 2002 ) ; slovaks } ; settlement } ; pivnice } } = true"
	tree = ASTTree.from_logic_str(logic_str=logic_str)
	tree.print_graph()
	gen_str = tree.to_logic_str()
	gen_str = add_sep_spaces(gen_str)
	gen_str += "= true"
	assert logic_str == gen_str


if __name__ == '__main__':
	test_to_str()