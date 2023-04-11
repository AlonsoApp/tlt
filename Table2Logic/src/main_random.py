from tqdm import tqdm

from config import read_arguments_train, write_config_to_file
from data_loader import get_data_loader
from intermediate_representation.lf_parser import ASTTree, get_cased_values
from logic2text import dataset_utils
from model.example import Example
from utils import create_experiment_folder
from logic2text.utils import flatten

from intermediate_representation import lf_grammar
import random
from logic2text.generation.random_lf_generator import aggregation, comparative, count, majority, ordinal, superlative, unique

actions_fn = {"aggregation": aggregation, "comparative": comparative, "count": count,
              "majority": majority, "ordinal": ordinal, "superlative": superlative, "unique": unique}

def generate_random_lf(example: Example) -> ASTTree:
    random_logic_node = random.choice(list(actions_fn.values()))(example.pd_table)
    random_logic = random_logic_node.todic()['tostr']
    cased_values = get_cased_values(random_logic)
    cased_values["case1a"] = flatten(example.table)
    astTree = ASTTree.from_logic_str(logic_str=random_logic, columns=example.column_names, indexes=example.indexes, cased_values=cased_values)
    return astTree


def execute():
    args = read_arguments_train()

    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
    print("Run experiment '{}'".format(experiment_name))

    write_config_to_file(args, output_path)

    val_data = dataset_utils.load_dataset(args.data_dir, "valid.json", use_small=args.toy)
    val_loader = get_data_loader(val_data, args.batch_size, False)

    grammar = lf_grammar.Grammar()

    sketch_correct, rule_label_correct, not_all_values_found, total = 0, 0, 0, 0
    predictions = []
    logically_correct_per_batch = []
    for batch in tqdm(val_loader, desc="Evaluating"):

        for data_row in batch:
            try:
                example = Example(data_row)
            except Exception as e:
                print("Exception while building example (evaluation): {}".format(e))
                continue

            random_lf = generate_random_lf(example)

            try:
                # here we set assemble the predicted actions (including leaf-nodes) as string
                full_prediction = " ".join([str(x) for x in random_lf.to_action_list()])
            except Exception as e:
                # print(e)
                full_prediction = ""

            prediction = {}

            # here we set assemble the predicted sketch actions as string
            prediction['sketch_result'] = " ".join(str(x) for x in random_lf.get_sketch())
            prediction['model_result'] = full_prediction

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.tgt_actions])

            if truth_sketch == prediction['sketch_result']:
                sketch_correct += 1
            if truth_rule_label == prediction['model_result']:
                rule_label_correct += 1

            total += 1

            predictions.append(prediction)

    sketch_acc, acc = float(sketch_correct) / float(total), float(rule_label_correct) / float(total)

    print("sketch_acc: {}".format(sketch_acc))
    print("acc: {}".format(acc))

if __name__ == '__main__':
    finished = False
    while finished is False:
        try:
            execute()
            finished = True
        except:
            None

