import copy

from typing import List

import torch

from intermediate_representation.lf_parser import ASTTree
from intermediate_representation.lf_grammar import Grammar
import neural_network_utils as nn_utils

from logic2text.utils import flatten
from intermediate_representation.lf_parser import OOV_TOKEN


class Example:
    def __init__(self, sample_dict, args, schema_len=None):#, value_cases_in_extra="case1b;case2;case3", masked_cases="", include_oov_token=False):
        """
        topic: ['2011', 'british', 'gt', 'season']
        table: [['row 0', '1', 'oulton park', '25 april', '60 mins', 'no 5 scuderia vittoria', 'no 1 trackspeed', 'no 44 abg motorsport'], ['row 1', '1', 'oulton park', ...
        table_len: 152
        column_names: ['round', 'circuit', 'date', 'length', 'pole position', 'gt3 winner', 'gt4 winner']
        col_num: 7
        values: ['12'] #case1b + case2 + case3 values
        indexes: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
        logic_str: 'eq { count { filter_eq { all_rows ; length ; 60 mins } } ; 12 } = true'
        schema_len: None ??
        tgt_actions: [Stat(4), Obj(3), N(0), View(1), View(0), C(3), Obj(4), V(3), Obj(4), V(152)]
        truth_actions: [Stat(4), Obj(3), N(0), View(1), View(0), C(3), Obj(4), V(3), Obj(4), V(152)]
        sketch: [Stat(4), Obj(3), N(0), View(1), View(0), Obj(4), Obj(4)]
        pd_table: pandas df of the table
        """
        self.topic = sample_dict["topic"]
        self.table = sample_dict["table"]
        self.table_len = sum([len(row) for row in self.table])
        self.column_names = sample_dict["columns"]
        self.col_num = len(self.column_names)
        cased_values = sample_dict["cased_values"]
        masked_cases = args.masked_cases.split(";")
        self.values_extra = build_values_extra(cased_values, args.value_cases_in_extra, masked_cases)  # values (case 1b, 2 and 3)
        #self.indexes = [str(x) for x in range(1, len(self.table) + 1)]
        # We use the fixed 20 indexes per sample because if the model can choose between 1 and len(table) it might choose an
        # incorrect id_c and break during conversion from action list to ASTTree
        self.indexes = [str(x) for x in range(1, args.max_index + 1)]
        self.indexes_len = len(self.indexes)

        self.logic_str = sample_dict["logic_str"]
        self.schema_len = schema_len

        # we replace all case 1 values found in gold LF with the entire table values including 'row N' tokens
        cased_values_with_table = cased_values.copy()
        cased_values_with_table["case1a"] = flatten(self.table)
        # all the values form which the pointer network chooses, including (if set in config) the oov_token
        self.pointer_values = cased_values_with_table["case1a"] + self.values_extra  # WATCH OUT! What happens if case1a is masked?
        if args.include_oov_token:
            self.pointer_values.append(OOV_TOKEN)
        self.pointer_values_len = len(self.pointer_values)
        self.ast_tree = ASTTree.from_logic_str(logic_str=self.logic_str,
                                               columns=self.column_names,
                                               indexes=self.indexes,
                                               cased_values=cased_values_with_table,
                                               masked_cases=masked_cases)
        self.tgt_actions = self.ast_tree.to_action_list()
        truth_actions = copy.deepcopy(self.tgt_actions)

        self.sketch = list()
        if truth_actions:
            for ta in truth_actions:
                if type(ta) not in Grammar.terminal_actions():
                    self.sketch.append(ta)
        self.pd_table = sample_dict["pd_table"]

def build_values_extra(cased_values, value_cases_in_extra, masked_cases):
    values_extra = []
    if value_cases_in_extra == '':
        return []
    for case in value_cases_in_extra.split(";"):
        if case not in masked_cases:
            # if the value is masked it should not be included in the extra list
            values_extra += cased_values[case]
    return values_extra


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Batch(object):
    """
    topic: ['2011', 'british', 'gt', 'season']
    table: [['row 0', '1', 'oulton park', '25 april', '60 mins', 'no 5 scuderia vittoria', 'no 1 trackspeed', 'no 44 abg motorsport'], ['row 1', '1', 'oulton park', ...
    table_len: 152
    column_names: ['round', 'circuit', 'date', 'length', 'pole position', 'gt3 winner', 'gt4 winner']
    col_num: 7
    values: ['12'] #case2 + case3 values
    indexes: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    logic_str: 'eq { count { filter_eq { all_rows ; length ; 60 mins } } ; 12 } = true'
    schema_len: None ??
    tgt_actions: [Stat(4), Obj(3), N(0), View(1), View(0), C(3), Obj(4), V(3), Obj(4), V(152)]
    truth_actions: [Stat(4), Obj(3), N(0), View(1), View(0), C(3), Obj(4), V(3), Obj(4), V(152)]
    sketch: [Stat(4), Obj(3), N(0), View(1), View(0), Obj(4), Obj(4)]

    batch.src_sents,
    batch.table_sents,
    batch.table_names,
    batch.values

    topics, tables, column_names, values
    """
    def __init__(self, examples: List[Example], grammar, cuda=False):
        self.examples = examples

        if examples[0].tgt_actions:
            self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
            self.max_sketch_num = max(len(e.sketch) for e in self.examples)

        self.topics = [e.topic for e in self.examples]
        self.topic_len = [len(e.topic) for e in self.examples]

        self.tables = [e.table for e in self.examples]
        self.table_len = [e.table_len for e in self.examples]
        self.column_names = [e.column_names for e in self.examples]
        self.col_num = [e.col_num for e in self.examples]

        self.indexes = [e.indexes for e in self.examples]
        self.indexes_len = [e.indexes_len for e in self.examples]

        self.values_extra = [e.values_extra for e in examples]
        self.n_values_extra = [len(e.values_extra) for e in examples]

        self.pointer_values = [e.pointer_values for e in examples]
        self.pointer_values_len = [e.pointer_values_len for e in examples]

        self.grammar = grammar
        self.cuda = cuda

    def __len__(self):
        return len(self.examples)

    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)

    @cached_property
    def pointer_value_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.pointer_values_len, cuda=self.cuda)

    @cached_property
    def index_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.indexes_len, cuda=self.cuda, max_len=20)

    @cached_property
    def column_token_mask(self):
        # we use this to mask paddings
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def topic_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.topic_len, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        # we use this to mask paddings
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def src_token_mask(self):
        """
        generates a mask where 0 correspondos to a position whereatoken of topic or table would stand in de concatenated
        tensor of topic + table. As the concatenation occurs after padding, the resulting tensor would be similar to
        this [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1] being 1 the padding tokens stand
        :return:
        """
        return torch.cat((self.topic_token_mask, self.table_token_mask), 1)

