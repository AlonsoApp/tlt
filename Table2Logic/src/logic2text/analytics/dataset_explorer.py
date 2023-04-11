import json
from tabulate import tabulate

from logic2text.utils import build_table, build_tree

if __name__ == '__main__':
    dataset_path = "./inferences/output_logic2text/t2l/3/dataset/test.json"
    with open(dataset_path) as f:
        data_in = json.load(f)
    hashed_dataset = {}
    for sample in data_in:
        hashed_dataset[sample["sha1"]] = sample
    while True:
        query_sha1 = input("Sha1: ")
        sample = hashed_dataset[query_sha1]
        pd_table = build_table(sample)
        ast_tree = build_tree(sample, pd_table)
        print("Topic: {}".format(sample["topic"]))
        print(tabulate(pd_table, headers='keys', tablefmt='psql'))
        ast_tree.print_graph()
        print(ast_tree.execute(pd_table))


