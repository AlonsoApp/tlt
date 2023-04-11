import random
from pandas.api.types import is_datetime64_any_dtype
from scipy.stats import norm

from execution.APIs import *

# Helpers
def filter_criterion_value(table:pd.DataFrame, column:str, criterion:str):
    def eq(table:pd.DataFrame, column:str):
        return random.choice(table[column].values)
    def greater(table:pd.DataFrame, column:str):
        values = table[column].values.tolist()
        max_val = values.pop(values.index(max(values)))
        return random.choice(values) if len(values)>0 else max_val
    def less(table:pd.DataFrame, column:str):
        values = table[column].values.tolist()
        min_val = values.pop(values.index(min(values)))
        return random.choice(values) if len(values)>0 else min_val
    def all(table:pd.DataFrame, column:str):
        return None

    options = {"filter_eq": eq,
               "filter_str_eq": eq,
               "filter_str_not_eq": eq,
               "filter_not_eq": eq,
               "filter_greater": greater,
               "filter_greater_eq": eq,
               "filter_less": less,
               "filter_less_eq": eq,
               "filter_all": all}

    return options[criterion](table, column)

def cols_where_row_has_unique_vals(table:pd.DataFrame, row_index:int):
    """
    Saca las cols que tiene un valor único en toda la tabla y puede utilizarse como id. If list [] returns all cols
    I asume there's at least one that makes it unique
    """
    row = table.iloc[row_index]
    colums = table.columns.tolist()
    #colums.remove(col_not_to_use_as_id)
    selectable_cols = []
    for col in colums:
        row_val = row[col]
        if len(table[table[col] == row_val]) == 1:
            selectable_cols.append(col)
    return selectable_cols

def scope_unique_two(table:pd.DataFrame, row_a_index:int, row_b_index:int, col_not_to_use_as_id:str):
    """
    Devuelve los nodos de scope de identifican unequivocamente dos rows bajo la misma columna (a ser posible)
    si no es posible devuelve cada uno con la col y valor que identifican unequvicamente la row en el index especificado
    """
    unique_cols_a = cols_where_row_has_unique_vals(table, row_a_index)
    unique_cols_a = table.columns.tolist() if len(unique_cols_a) == 0 else unique_cols_a
    unique_cols_b = cols_where_row_has_unique_vals(table, row_b_index)
    unique_cols_b = table.columns.tolist() if len(unique_cols_b) == 0 else unique_cols_b
    unique_cols_ab = list(set(unique_cols_a).intersection(unique_cols_b))
    # let's try not to use the col we are talking about
    if len(unique_cols_ab) > 1 and col_not_to_use_as_id in unique_cols_ab:
        unique_cols_ab.remove(col_not_to_use_as_id)
    # pray so there's at least one
    if len(unique_cols_ab)>0:
        column_scope_a = unique_cols_ab[0]
        column_scope_b = unique_cols_ab[0]
    else:
        column_scope_a = unique_cols_a[0]
        column_scope_b = unique_cols_b[0]
    scope_a = Node(table, "filter_str_eq", ["all_rows", column_scope_a, table[column_scope_a][row_a_index]])
    scope_b = Node(table, "filter_str_eq", ["all_rows", column_scope_b, table[column_scope_b][row_b_index]])
    return scope_a, scope_b

def filter_scope(table:pd.DataFrame, previous_scope="all_rows"):
    if isinstance(previous_scope, Node):
        table = previous_scope.result()
    # Based on which column, does the statement choose the subset?
    columns = list(table.columns)
    column = random.choice(columns)
    # Select the criterion, based on which we filter the table records to select this subset.
    if is_numericable(table, column):
        filter_criterion = random.choice(["filter_eq", "filter_not_eq", "filter_greater", "filter_greater_eq", "filter_less", "filter_less_eq"])
    else:
        filter_criterion = random.choice(["filter_str_eq", "filter_str_not_eq"])
    #  Write the value to be filtered for selection of this subset
    value_to_filter = filter_criterion_value(table, column, filter_criterion)

    scope_node = Node(table, filter_criterion, [previous_scope, column, value_to_filter])
    if len(scope_node.result()) == 0:
        # give it another try, I know it's not the right way but let's leave it as a TODO find a way to avoid empty scope
        return filter_scope(table, previous_scope)
    return scope_node

def is_numericable(table:pd.DataFrame, column:str):
    if is_numeric(table, column) or is_datetime64_any_dtype(table[column]):
        return True
    else:
        pats = table[column].str.extract(pat_add, expand=False)
        if pats.isnull().all():
            pats = table[column].str.extract(pat_num, expand=False)
        if pats.isnull().all():
            return False
        return True

def is_numeric(table:pd.DataFrame, column:str):
    return column in get_numeric_cols(table)

def get_numeric_cols(table:pd.DataFrame):
    num_table = table.copy()
    for c in table.columns:
        try:
            num_table[c] = pd.to_numeric(table[c])
        except:
            pass
    return num_table.select_dtypes(include=[np.number]).columns

def to_numeric(s):
    s = s.replace(" ", "")
    try:
        return int(s)
    except ValueError:
        return float(s)

def get_numericable_cols(table:pd.DataFrame):
    """
    Gets the numeric columns + the str columns that could be interpreted as numeric
    """
    num_cols = get_numeric_cols(table).tolist()
    rest_cols = list(set(num_cols) ^ set(table.columns))
    return num_cols + [col for col in rest_cols if is_numericable(table, col)]

def comparative_func(a, b, is_col_numeric:bool):
    """
    Gets a comparative func that satisfies the difference between the values a and b
    """
    index = 0 if is_col_numeric else 1
    if APIs[["eq","str_eq"][index]]["function"](a, b):
        return ["eq","str_eq"][index]
    elif random.choice([True, False]) and is_col_numeric:
        return "diff"
    elif APIs["greater"]["function"](a, b):
        return "greater"
    elif APIs["less"]["function"](a, b):
        return "less"
    elif APIs["eq"]["function"](a, b):
        # Last chance, eq is less restrictive
        return "eq"
    return None

def change_dtype_pure_numeric(table:pd.DataFrame):
    for c in table.columns:
        try:
            table[c] = pd.to_numeric(table[c])
        except:
            pass
    return table

def most_common(lst):
    return max(set(lst), key=lst.count)

def random_with_distribution(n_min, n_max, mu, sigma):
    return min(n_max, max(n_min, random.gauss(mu, sigma)))

def valid_majority_functions(table:pd.DataFrame, column:str, candidate):
    funcs = ["all_str_eq", "all_str_not_eq", "all_eq", "all_not_eq", "all_less", "all_less_eq", "all_greater",
             "all_greater_eq", "most_str_eq", "most_str_not_eq", "most_eq", "most_not_eq", "most_less", "most_less_eq",
             "most_greater", "most_greater_eq"]
    executed_funcs = {}
    for func in funcs:
        executed_funcs[func] = APIs[func]["function"](table, column, candidate)
    return [fn_name for fn_name, is_valid in executed_funcs.items() if is_valid]

def get_mentioned_columns(table:pd.DataFrame, cols_to_avoid:[str]):
    mentioned_cols = []
    mentionable_cols = table.columns.tolist()
    mentionable_cols = [col for col in mentionable_cols if col not in cols_to_avoid]
    sel_prob = 0.4
    for col in mentionable_cols:
        if np.random.choice([True, False], p=[sel_prob, 1 - sel_prob]):
            mentioned_cols.append(col)
            sel_prob = sel_prob / 2
    return mentioned_cols

def random_unique_col_val(table:pd.DataFrame):
    row_indexes = [x for x in range(0, len(table))]
    random.shuffle(row_indexes)
    for index in row_indexes:
        cols = cols_where_row_has_unique_vals(table, index)
        if len(cols) > 0:
            col = random.choice(cols)
            return col, table[col].iloc[index]

# Generation functions
def aggregation(table:pd.DataFrame):
    # Choose whether the aggregation is performed on the scope of all table rows, or on a subset of all rows:
    use_scope = random.choice([True, False])
    scope = filter_scope(table) if use_scope else "all_rows"
    # What is the table column that the superlative action is performed on?
    # We get only numeric columns
    columns = get_numericable_cols(table)
    column = random.choice(columns)
    # What is the type of this aggregation, sum or average?
    agg_type = random.choice(["sum", "avg"])
    agg_node = Node(table, agg_type, [scope, column])

    # What is the result of this aggregation?
    result = "{0:.2f}".format(agg_node.result())
    final_node = Node(table, "round_eq", [result, agg_node])
    return final_node

def comparative(table:pd.DataFrame):
    # We get only numeric columns
    numeric_cols = get_numericable_cols(table)
    # Which column is the statement comparing?
    col_to_compare = random.choice(numeric_cols)
    # What is the first row to be compared?
    row_a_index = random.randint(0, len(table.index)-1)
    row_b_index = random.choice([x for x in range(len(table.index)) if x != row_a_index])
    # What is the relationship comparing the records numerically in the first row with the second?
    # Is the compared records itself mentioned in the statement?
    # en función de esto sacamos los valores a comparar o no
    is_mentioned = random.choice([True, False])

    scope_a, scope_b = scope_unique_two(table, row_a_index, row_b_index, col_to_compare)
    is_col_pure_numeric = is_numeric(table, col_to_compare)
    hop_op = 'num_hop' if is_col_pure_numeric else 'str_hop'
    hop_node_a = Node(table, hop_op, [scope_a, col_to_compare])
    hop_node_b = Node(table, hop_op, [scope_b, col_to_compare])
    result_a =  hop_node_a.result()
    result_b = hop_node_b.result()

    comp_func = comparative_func(result_a, result_b, is_col_pure_numeric)
    comp_node = Node(table, comp_func, [hop_node_a, hop_node_b])
    if comp_func == "diff":
        comp_node = Node(table, "eq", [comp_node.result(), comp_node])

    if not is_mentioned:
        return comp_node

    result_comp_op = "eq" if is_col_pure_numeric else "str_eq"
    result_node_a = Node(table, result_comp_op, [result_a, hop_node_a])
    result_node_b = Node(table, result_comp_op, [result_b, hop_node_b])
    mention_node = Node(table, "and", [result_node_a, result_node_b])
    return Node(table, "and", [mention_node, comp_node])

def count(table:pd.DataFrame):
    # Select the scope of the counting
    scope = filter_scope(table)
    use_scope = np.random.choice([True, False], p=[.15, .85])
    if use_scope and len(scope.result())>0:
        scope = filter_scope(table, scope)
    count_node = Node(table, "count", [scope])
    # Write the value to be filtered for counting
    result = str(count_node.result())
    final_node = Node(table, "eq", [result, count_node])
    return final_node

def majority(table:pd.DataFrame):
    # What is the scope of this majority statement?
    use_scope = np.random.choice([True, False], p=[.2, .8])
    scope = filter_scope(table) if use_scope else "all_rows"
    # Which column the statement is describing?
    column = random.choice(table.columns)
    scope_table = scope.result() if isinstance(scope, Node) else table
    values = scope_table[column].values.tolist()
    # Is the statement describing all the records or most frequent records within the scope?
    mcv = most_common(values)
    candidates = [mcv]
    if is_numeric(table, column) and len(set(values))>1:
        values = [to_numeric(val) for val in values]
        mu, std = norm.fit(values)
        candidates.append(str(random_with_distribution(min(values), max(values), mu, std)))
    candidate = random.choice(candidates)

    valid_funcs = valid_majority_functions(scope_table, column, candidate)
    # if our random number didn't get any func=True we try with the mcv (most common value)
    if len(valid_funcs)==0:
        valid_funcs = valid_majority_functions(table, column, mcv)
        candidate = mcv

    final_node = Node(table, random.choice(valid_funcs), [scope, column, candidate])
    return final_node

def ordinal(table:pd.DataFrame):
    # What is the scope of this majority statement?
    use_scope = np.random.choice([True, False], p=[.1, .9])
    scope = filter_scope(table) if use_scope else "all_rows"
    # What is the table column that the ordinal description is based on?
    numeric_cols = get_numeric_cols(table)
    column = random.choice(numeric_cols) if len(numeric_cols)>0 else random.choice(get_numericable_cols(table))
    # Is the ordinal description based on a numerically max to min or min to max ranking of the column records?
    ordinal_base = random.choice(["max-min","min-max"])
    # What is the table row containing this n-th record?
    scope_table = scope.result() if isinstance(scope, Node) else table
    row_index = random.randint(0, len(scope_table.index) - 1)

    # On this row, what are the other column(s) mentioned? (not including the column describing scope) If not any other column is mentioned, write 'n/a'.
    mentioned_other_cols = get_mentioned_columns(table, [column])
    total_mention_other_node = None
    for col in mentioned_other_cols:
        func = "nth_argmax" if ordinal_base == "max-min" else "nth_argmin"
        ordinal_node = Node(table, func, [scope, column, str(row_index+1)])
        hop_fn = "num_hop" if is_numericable(table, col) else "str_hop"
        hop_node = Node(table, hop_fn, [ordinal_node, col])
        eq_fn = "eq" if is_numericable(table, col) else "str_eq"
        result = str(hop_node.result())
        mention_other_node = Node(table, eq_fn, [result, hop_node])
        if total_mention_other_node is None:
            total_mention_other_node = mention_other_node
        else:
            total_mention_other_node = Node(table, "and", [total_mention_other_node, mention_other_node])
    # Is this n-th record itself mentioned in the statement?
    is_mentioned = True if total_mention_other_node is None else np.random.choice([True, False], p=[.1,.9])
    if is_mentioned:
        func = "nth_max" if ordinal_base == "max-min" else "nth_min"
        ordinal_node = Node(table, func, [scope, column, str(row_index+1)])
        result = str(ordinal_node.result())
        mention_node = Node(table, "eq", [result, ordinal_node])
        if total_mention_other_node is None:
            total_mention_other_node = mention_node
        else:
            total_mention_other_node = Node(table, "and", [total_mention_other_node, mention_node])
    return total_mention_other_node

def superlative(table:pd.DataFrame):
    # Is the superlative action performed on the scope of all table rows, or on a subset of all rows?
    use_scope = np.random.choice([True, False], p=[.1, .9])
    scope = filter_scope(table) if use_scope else "all_rows"
    # What is the table column that the superlative action is performed on?
    column = random.choice(get_numericable_cols(table))
    # Is the superlative action taking the numerical maximum, or minimum value among the records? [max/min]
    superlative_base = random.choice(["max","min"])
    # What is the table row containing this superlative value?
    # On this row with the superlative value, what are the other column(s) mentioned?
    mentioned_other_cols = get_mentioned_columns(table, [column])
    total_mention_other_node = None
    for col in mentioned_other_cols:
        func = "argmax" if superlative_base == "max" else "argmin"
        superlative_node = Node(table, func, [scope, column])
        hop_fn = "num_hop" if is_numericable(table, col) else "str_hop"
        hop_node = Node(table, hop_fn, [superlative_node, col])
        eq_fn = "eq" if is_numericable(table, col) else "str_eq"
        result = str(hop_node.result())
        mention_other_node = Node(table, eq_fn, [result, hop_node])
        if total_mention_other_node is None:
            total_mention_other_node = mention_other_node
        else:
            total_mention_other_node = Node(table, "and", [total_mention_other_node, mention_other_node])
    # Is this n-th record itself mentioned in the statement?
    is_mentioned = True if total_mention_other_node is None else np.random.choice([True, False], p=[.1, .9])
    if is_mentioned:
        func = "max" if superlative_base == "max" else "min"
        superlative_node = Node(table, func, [scope, column])
        result = str(superlative_node.result())
        mention_node = Node(table, "eq", [result, superlative_node])
        if total_mention_other_node is None:
            total_mention_other_node = mention_node
        else:
            total_mention_other_node = Node(table, "and", [total_mention_other_node, mention_node])
    return total_mention_other_node

def unique(table:pd.DataFrame):
    # What is the scope of this statement describing unique row?
    use_scope = np.random.choice([True, False], p=[.1, .9])
    scope = filter_scope(table) if use_scope else "all_rows"
    # What is this unique row?
    # Write the table column that shows the uniqueness of this row
    # Based on the selected criterion, write the value to be filtered for the unqiue row.
    scope_table = scope.result() if isinstance(scope, Node) else table
    column, value = random_unique_col_val(scope_table)
    # Select the criterion, based on which we filter records in this column to find the unique row.
    criterion = "filter_eq" if is_numeric(scope_table, column) else "filter_str_eq"
    filter_node = Node(table, criterion, [scope, column, value])
    unique_node = Node(table, "only", [filter_node])
    # On this unique row, what are the other column(s) mentioned (except the column describing the scope)?
    mentioned_other_cols = get_mentioned_columns(table, [column])
    total_mention_other_node = None
    for col in mentioned_other_cols:
        hop_fn = "num_hop" if is_numericable(table, col) else "str_hop"
        hop_node = Node(table, hop_fn, [filter_node, col])
        eq_fn = "eq" if is_numeric(table, col) else "str_eq"
        result = str(hop_node.result())
        mention_other_node = Node(table, eq_fn, [result, hop_node])
        if total_mention_other_node is None:
            total_mention_other_node = mention_other_node
        else:
            total_mention_other_node = Node(table, "and", [total_mention_other_node, mention_other_node])
    if total_mention_other_node is not None:
        total_mention_other_node = Node(table, "and", [total_mention_other_node, unique_node])
    else:
        total_mention_other_node = unique_node
    return total_mention_other_node


class Node(object):
    def __init__(self, table: pd.DataFrame, func: str, args: []=None):
        self.table = table
        self.func = func
        if args is None:
            args = []
        self.args = args
        indexes = [0]
        for arg in args:
            if isinstance(arg, Node):
                indexes.append(arg.ind)
        self.ind = max(indexes)+1

    def todic(self):
        dic_args = []
        for arg in self.args:
            if isinstance(arg, Node):
                dic_args.append(arg.todic())
            else:
                dic_args.append(arg)
        return {"func":self.func, "args":dic_args, "result":True, "ind":self.ind, "tostr":self.tostr(), "tointer":""}

    def tostr(self):
        str_args = []
        for arg in self.args:
            if isinstance(arg, Node):
                str_args.append(arg.tostr())
            else:
                str_args.append(arg)
        return APIs[self.func]["tostr"](*str_args)

    def result(self):
        args = []
        for arg in self.args:
            if isinstance(arg, Node):
                args.append(arg.result())
            else:
                if arg == "all_rows":
                    args.append(self.table)
                else:
                    args.append(arg)
        result = APIs[self.func]["function"](*args)
        if APIs[self.func]["output"] == "str":
            result = str(result)
        return result