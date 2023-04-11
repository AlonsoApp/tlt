r"""
Given any json of the dataset shows the amount of False logically executed LFs
"""

import json

from logic2text.utils import build_table, build_tree

if __name__ == '__main__':
    dataset_path = "./inferences/exp_2/test.json"
    with open(dataset_path) as f:
        data_in = json.load(f)
    hashed_dataset = {}
    count_false = 0
    for sample in data_in:
        pd_table = build_table(sample)
        ast_tree = build_tree(sample, pd_table)
        if not ast_tree.execute(pd_table):
            count_false +=1

    print("False: {} out of {}".format(count_false, len(data_in)))


