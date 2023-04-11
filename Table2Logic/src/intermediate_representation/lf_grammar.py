keywords = ['only', 'and', 'greater', 'less', 'eq', 'str_eq', 'not_eq', 'not_str_eq', 'round_eq', 'all_eq',
            'all_str_eq', 'all_not_eq', 'all_str_not_eq', 'all_less', 'all_less_eq', 'all_greater', 'all_greater_eq',
            'most_eq', 'most_str_eq', 'most_not_eq', 'most_str_not_eq', 'most_less', 'most_less_eq', 'most_greater',
            'most_greater_eq', 'all_rows', 'filter_eq', 'filter_str_eq', 'filter_not_eq', 'filter_str_not_eq',
            'filter_less', 'filter_greater', 'filter_greater_eq', 'filter_less_eq', 'filter_all', 'count', 'avg', 'sum',
            'max', 'min', 'nth_max', 'nth_min', 'argmax', 'argmin', 'nth_argmax', 'nth_argmin', 'str_hop', 'num_hop',
            'str_hop_first', 'num_hop_first', 'diff']#, 'hop', 'hop_first']


class Grammar(object):
    def __init__(self, is_sketch=False):
        self.begin = 0
        self.type_id = 0
        self.is_sketch = is_sketch
        self.prod2id = {}
        self.type2id = {}
        self._init_grammar(Stat)
        self._init_grammar(View)
        self._init_grammar(N)
        self._init_grammar(Row)
        self._init_grammar(Obj)

        self._init_id2prod()
        self.type2id[C] = self.type_id
        self.type_id += 1
        self.type2id[I] = self.type_id
        self.type_id += 1
        self.type2id[V] = self.type_id

    def _init_grammar(self, Cls):
        """
        get the production of class Cls
        :param Cls:
        :return:
        """
        production = Cls._init_grammar()
        for p in production:
            self.prod2id[p] = self.begin
            self.begin += 1
        self.type2id[Cls] = self.type_id
        self.type_id += 1

    def _init_id2prod(self):
        self.id2prod = {}
        for key, value in self.prod2id.items():
            self.id2prod[value] = key

    def get_production(self, Cls):
        return Cls._init_grammar()

    @staticmethod
    def terminal_actions():
        return [C, I, V]

    @staticmethod
    def sketch_actions():
        return [Stat, View, N, Row, Obj]


class Action(object):
    def __init__(self):
        self.pt = 0
        self.production = None
        self.children = list()

    def get_next_action(self, is_sketch=False):
        """
        Gets the actions present in the production rule of this action
        :param is_sketch:
        :return: a list of actions in this production rule
        """
        actions = list()
        for x in self.production.split(' ')[1:]:
            if x not in keywords:
                rule_type = eval(x)
                if is_sketch:
                    # if we are processing the sketch we don't return leave nodes
                    if rule_type not in [C, I, V]:
                        actions.append(rule_type)
                else:
                    actions.append(rule_type)
        return actions

    def set_parent(self, parent):
        self.parent = parent

    def add_children(self, child):
        self.children.append(child)

    def potential_case_value(self):
        return 0


class Stat(Action):
    id_only = 0
    id_and = 1
    id_greater = 2
    id_less = 3
    id_eq = 4
    id_not_eq = 5
    id_round_eq = 6
    id_all_eq = 7
    id_all_not_eq = 8
    id_all_less = 9
    id_all_less_eq = 10
    id_all_greater = 11
    id_all_greater_eq = 12
    id_most_eq = 13
    id_most_not_eq = 14
    id_most_less = 15
    id_most_less_eq = 16
    id_most_greater = 17
    id_most_greater_eq = 18
    id_str_eq = 19
    id_not_str_eq = 20
    id_all_str_eq = 21
    id_all_str_not_eq = 22
    id_most_str_eq = 23
    id_most_str_not_eq = 24

    def __init__(self, id_c, parent=None):
        super(Stat, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(cls):
        cls.grammar_dict = {
            Stat.id_only: 'Stat only View',
            Stat.id_and: 'Stat and Stat Stat',
            Stat.id_greater: 'Stat greater Obj Obj',  # case 3
            Stat.id_less: 'Stat less Obj Obj',  # case 3
            Stat.id_eq: 'Stat eq Obj Obj',  # -1 analyze sibling
            Stat.id_not_eq: 'Stat not_eq Obj Obj',  # case 3
            Stat.id_round_eq: 'Stat round_eq Obj Obj',  # -1 analyze sibling
            Stat.id_all_eq: 'Stat all_eq View C Obj',  # case 4
            Stat.id_all_not_eq: 'Stat all_not_eq View C Obj',  # case 3
            Stat.id_all_less: 'Stat all_less View C Obj',  # case 3
            Stat.id_all_less_eq: 'Stat all_less_eq View C Obj',  # case 3
            Stat.id_all_greater: 'Stat all_greater View C Obj',  # case 3
            Stat.id_all_greater_eq: 'Stat all_greater_eq View C Obj',  # case 3
            Stat.id_most_eq: 'Stat most_eq View C Obj',  # case 4
            Stat.id_most_not_eq: 'Stat most_not_eq View C Obj',  # case 3
            Stat.id_most_less: 'Stat most_less View C Obj',  # case 3
            Stat.id_most_less_eq: 'Stat most_less_eq View C Obj',  # case 3
            Stat.id_most_greater: 'Stat most_greater View C Obj',  # case 3
            Stat.id_most_greater_eq: 'Stat most_greater_eq View C Obj',  # case 3
            Stat.id_str_eq: 'Stat str_eq Obj Obj',  # -1 analyze sibling
            Stat.id_not_str_eq: 'Stat not_str_eq Obj Obj',  # case 3
            Stat.id_all_str_eq: 'Stat all_str_eq View C Obj',  # case 4
            Stat.id_all_str_not_eq: 'Stat all_str_not_eq View C Obj',  # case 3
            Stat.id_most_str_eq: 'Stat most_str_eq View C Obj',  # case 4
            Stat.id_most_str_not_eq: 'Stat most_str_not_eq View C Obj',  # case 3
        }
        cls.production_id = {}
        for id_x, value in enumerate(cls.grammar_dict.values()):
            cls.production_id[value] = id_x

        return cls.grammar_dict.values()

    def __str__(self):
        return 'Stat(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Stat(' + str(self.id_c) + ')'

    def potential_case_value(self):
        if self.id_c in [Stat.id_greater, Stat.id_less, Stat.id_all_less, Stat.id_all_less_eq,
                         Stat.id_all_greater, Stat.id_all_greater_eq, Stat.id_most_less, Stat.id_most_less_eq,
                         Stat.id_most_greater, Stat.id_most_greater_eq, Stat.id_not_eq, Stat.id_not_str_eq,
                         Stat.id_all_not_eq, Stat.id_all_str_not_eq, Stat.id_most_not_eq,
                         Stat.id_most_str_not_eq]:
            return 3
        elif self.id_c in [Stat.id_all_eq, Stat.id_most_eq, Stat.id_all_str_eq, Stat.id_most_str_eq]:
            return 4
        elif self.id_c in [Stat.id_eq, Stat.id_round_eq, Stat.id_str_eq]:
            return -1 # search deeper, look siblings
        else:
            return 0


class View(Action):
    id_all_rows = 0
    id_filter_eq = 1
    id_filter_not_eq = 2
    id_filter_less = 3
    id_filter_greater = 4
    id_filter_greater_eq = 5
    id_filter_less_eq = 6
    id_filter_all = 7
    id_filter_str_eq = 8
    id_filter_str_not_eq = 9

    def __init__(self, id_c, parent=None):
        super(View, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            View.id_all_rows: 'View all_rows',
            View.id_filter_eq: 'View filter_eq View C Obj',
            View.id_filter_not_eq: 'View filter_not_eq View C Obj',
            View.id_filter_less: 'View filter_less View C Obj',
            View.id_filter_greater: 'View filter_greater View C Obj',
            View.id_filter_greater_eq: 'View filter_greater_eq View C Obj',
            View.id_filter_less_eq: 'View filter_less_eq View C Obj',
            View.id_filter_all: 'View filter_all View C',
            View.id_filter_str_eq: 'View filter_str_eq View C Obj',
            View.id_filter_str_not_eq: 'View filter_str_not_eq View C Obj',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'View(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'View(' + str(self.id_c) + ')'

    def potential_case_value(self):
        if self.id_c in [View.id_filter_less, View.id_filter_greater, View.id_filter_greater_eq, View.id_filter_less_eq]:
            return 3
        elif self.id_c in [View.id_filter_eq, View.id_filter_not_eq, View.id_filter_str_eq, View.id_filter_str_not_eq]:
            return 4
        else:
            return 0


class N(Action):
    id_count = 0
    id_avg = 1
    id_sum = 2
    id_max = 3
    id_min = 4
    id_nth_max = 5
    id_nth_min = 6

    def __init__(self, id_c, parent=None):
        super(N, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            N.id_count: 'N count View',
            N.id_avg: 'N avg View C',
            N.id_sum: 'N sum View C',
            N.id_max: 'N max View C',
            N.id_min: 'N min View C',
            N.id_nth_max: 'N nth_max View C I',
            N.id_nth_min: 'N nth_min View C I'
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'N(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'N(' + str(self.id_c) + ')'

    def potential_case_value(self):
        # This time refers to the potential case of a sibling Value
        return 2 if self.id_c in [N.id_count, N.id_avg, N.id_sum] else 1


class Row(Action):
    id_argmax = 0
    id_argmin = 1
    id_nth_argmax = 2
    id_nth_argmin = 3

    def __init__(self, id_c, parent=None):
        super(Row, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            Row.id_argmax: 'Row argmax View C',
            Row.id_argmin: 'Row argmin View C',
            Row.id_nth_argmax: 'Row nth_argmax View C I',
            Row.id_nth_argmin: 'Row nth_argmin View C I'
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Row(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Row(' + str(self.id_c) + ')'


class Obj(Action):
    id_str_hop = 0
    id_num_hop = 1
    id_str_hop_first = 2
    id_num_hop_first = 3
    id_diff = 4
    id_N = 5
    id_V = 6

    def __init__(self, id_c, parent=None):
        super(Obj, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            Obj.id_str_hop: 'Obj str_hop Row C',
            Obj.id_num_hop: 'Obj num_hop Row C',
            Obj.id_str_hop_first: 'Obj str_hop_first View C',
            Obj.id_num_hop_first: 'Obj num_hop_first View C',
            Obj.id_diff: 'Obj diff Obj Obj',
            Obj.id_N: 'Obj N',
            Obj.id_V: 'Obj V',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Obj(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Obj(' + str(self.id_c) + ')'

    def potential_case_value(self):
        # This time refers to the potential case of a sibling Value. Any value compared to this Obj is...
        if self.id_c in [Obj.id_diff]:
            return 2
        elif self.id_c in [Obj.id_N]:
            return -1  # search deeper
        elif self.id_c in [Obj.id_num_hop, Obj.id_str_hop, Obj.id_num_hop_first, Obj.id_str_hop_first]:
            return 1
        elif self.id_c in [Obj.id_V]:
            return 3
        else:
            return 0


class C(Action):
    """
    Column.
    Ursin: A column, in contrary to a sketch-actions, has no grammar. The id_c will be the index of the column.
    """

    def __init__(self, id_c, parent=None):
        super(C, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self.production = 'C'
        self.table = None

    def __str__(self):
        return 'C(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'C(' + str(self.id_c) + ')'


class I(Action):
    """
    Index of a row
    Ursin: An index, in contrary to a sketch-actions, has no grammar. The id_c will be the index of the row.
    """

    def __init__(self, id_c, parent=None):
        super(I, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self.production = 'I'
        self.table = None

    def __str__(self):
        return 'I(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'I(' + str(self.id_c) + ')'


class V(Action):
    """
    This class represents a Value. It can be numeric (e.g. 15.25) or a string "Denmark".
    id_c is referring to the index of the value in the list of all possible values.
    """

    def __init__(self, id_c, parent=None):
        super(V, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self.production = 'V'
        self.table = None

    def __str__(self):
        return 'V(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'V(' + str(self.id_c) + ')'


if __name__ == '__main__':
    g = Grammar()
    print("Actions:")
    for key, value in g.type2id.items():
        print("Action: {}, id: {}".format(key, value))

    print()
    print()
    print("Production Rules:")

    for key, value in g.prod2id.items():
        print("Production Rule: {}, id: {}".format(key, value))
