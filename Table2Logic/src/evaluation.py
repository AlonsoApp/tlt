from typing import List

import torch
from tqdm import tqdm

from intermediate_representation.beam import Beams
from intermediate_representation.lf_parser import ASTTree
from model.example import Example

import traceback

def evaluate(model, dev_loader, args):
    model.eval()

    sketch_correct, rule_label_correct, not_all_values_found, total = 0, 0, 0, 0
    predictions = []
    for batch in tqdm(dev_loader, desc="Evaluating"):

        for data_row in batch:
            try:
                example = Example(data_row, args)
            except Exception as e:
                print("Exception while building example (evaluation): {}".format(e))
                continue

            with torch.no_grad():
                results_all = model.parse(example, beam_size=args.beam_size)
            results = results_all[0]
            result = select_result(results, example, args.rejection_sampling)
            try:
                # here we set assemble the predicted actions (including leaf-nodes) as string
                full_prediction = " ".join([str(x) for x in result.actions])
            except Exception as e:
                print(traceback.format_exc())
                full_prediction = ""

            prediction = {}

            # here we set assemble the predicted sketch actions as string
            prediction['sketch_result'] = " ".join(str(x) for x in results_all[1])
            prediction['model_result'] = full_prediction

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.tgt_actions])

            if truth_sketch == prediction['sketch_result']:
                sketch_correct += 1
            if truth_rule_label == prediction['model_result']:
                rule_label_correct += 1

            total += 1

            predictions.append(prediction)

    return float(sketch_correct) / float(total), float(rule_label_correct) / float(total), predictions

def select_result(results: List[Beams], sample: Example, rejection_sampling=False):
    """
    Given a resulting beam list, we execute every LF and return the correct most probable option. If no Lf is logically
    correct we return the most probable one
    :param results:
    :param sample:
    :param rejection_sampling: Whether we select the logically correct most probable one
    :return:
    """
    if len(results) == 0:
        return None, 0
    if rejection_sampling is False:
        # We cannot do RS if the generated LF contains [OOV] tokens
        return results[0]
    for result in results:
        tree = ASTTree.from_action_list(result.actions, columns=sample.column_names, indexes=sample.indexes, values=sample.pointer_values)
        exec_result = tree.execute(sample.pd_table)
        if exec_result is True:
            return result
    return results[0]