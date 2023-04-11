import json
import os

import torch
from tqdm import tqdm

from config import read_arguments_train, write_config_to_file
from data_loader import get_data_loader
from logic2text import dataset_utils
from model.model import IRNet
from inference import infer
from utils import setup_device, set_seed_everywhere, save_model, create_experiment_folder
from intermediate_representation.lf_parser import add_sep_spaces

from intermediate_representation import lf_grammar


def create_dataset_with_predictions(dataset_dir, dataset_file, output_dir, predictions):
    dataset_path = os.path.join(dataset_dir, dataset_file)
    with open(dataset_path) as f:
        data_in = json.load(f)
    assert len(predictions) == len(data_in)
    for i, data in enumerate(tqdm(data_in)):
        data["sent"] = "No text. LF generated with T2L model."
        data["annotation"] = None
        data["logic"] = {}  # no time to implement tree.to_logic_dic and isn't used to train Logic2Text
        data["logic_str"] = add_sep_spaces(predictions[i]) + "= true"
        data["interpret"] = "No interpret"

    with open(os.path.join(output_dir, dataset_file), 'w', encoding='utf-8') as f:
        json.dump(data_in, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = read_arguments_train()

    inference_name, output_path = create_experiment_folder(args.inference_output_dir, args.exp_name)
    print("Run inference '{}'".format(inference_name))

    write_config_to_file(args, output_path)

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    dataset = dataset_utils.load_dataset(args.data_dir, "test.json", use_small=args.toy)
    data_loader = get_data_loader(dataset, args.batch_size, False)

    grammar = lf_grammar.Grammar()
    model = IRNet(args, device, grammar)
    model.load_state_dict(torch.load(args.model_to_load_path))
    model.to(device)

    predictions = infer(model, data_loader, args)

    create_dataset_with_predictions(args.data_dir, "test.json", output_path, predictions)

