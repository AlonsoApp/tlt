import torch
from tqdm import tqdm

from intermediate_representation.lf_parser import ASTTree
from model.example import Example
from evaluation import select_result

def infer(model, data_loader, args):
    model.eval()

    predictions = []
    for batch in tqdm(data_loader, desc="Inferring"):

        for data_row in batch:
            try:
                example = Example(data_row, args)
            except Exception as e:
                print("Exception while building example (inference): {}".format(e))
                continue

            with torch.no_grad():
                results_all = model.parse(example, beam_size=args.beam_size)

            results = results_all[0]
            result = select_result(results, example, args.rejection_sampling)
            tree = ASTTree.from_action_list(result.actions, columns=example.column_names, indexes=example.indexes, values=example.pointer_values)
            logic_str = tree.to_logic_str()

            predictions.append(logic_str)

    return predictions
