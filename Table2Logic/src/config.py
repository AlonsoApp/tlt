import argparse
import json
import os


class Config:
    DATA_PREFIX = "data"
    EXPERIMENT_PREFIX = "experiments"
    INFERENCE_PREFIX = "inferences"


def write_config_to_file(args, output_path):
    config_path = os.path.join(output_path, "args.json")

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, indent=2)


def read_arguments_train():
    parser = argparse.ArgumentParser(description="Run training with following arguments")

    # general configuration
    parser.add_argument('--exp_name', default='exp', type=str)
    parser.add_argument('--seed', default=90, type=int)
    parser.add_argument('--toy', default=False, action='store_true')
    parser.add_argument('--data_set', default='Logic2Text/original_data_fix_tokens', type=str)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--cuda', action='store_true')

    # encoder configuration
    parser.add_argument('--encoder_pretrained_model', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_length', default=512, type=int)
    parser.add_argument('--value_cases_in_extra', default='case1b;case2;case3', type=str)

    parser.add_argument('--num_epochs', default=50.0, type=float)

    # training & optimizer configuration
    parser.add_argument('--lr_base', default=5e-5, type=float)  # lr_connection 1e-4 o 5e-5 (antes 1e-3)
    parser.add_argument('--lr_connection', default=1e-4, type=float)
    parser.add_argument('--lr_transformer', default=2e-5, type=float)
    # parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--scheduler_gamma', default=0.5, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--clip_grad', default=.1, type=float)  # Probar 0.1 (antes 5)
    parser.add_argument('--loss_epoch_threshold', default=50, type=int)
    parser.add_argument('--sketch_loss_weight', default=1.0, type=float)

    # model configuration
    parser.add_argument('--column_pointer', action='store_true', default=False)
    parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=300, type=int, help='size of LSTM hidden states')
    parser.add_argument('--action_embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--att_vec_size', default=300, type=int, help='size of attentional vector')
    parser.add_argument('--type_embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--col_embed_size', default=300, type=int, help='size of word embeddings')
    parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    parser.add_argument('--max_index', default=20, type=int, help='the largest possible index to choose by PointerNet while generating I')
    parser.add_argument('--index_embed_size', default=300, type=float, help='size of the embedding that represents each I')
    parser.add_argument('--include_oov_token', action='store_true', help='whether we include a token representing an OOV value. THIS MUST BE INCLUDED if we mask any value case (masked_cases != "")')

    # prediction configuration
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--decode_max_time_step', default=50, type=int,
                        help='maximum number of time steps used in decoding and sampling')
    parser.add_argument('--rejection_sampling', action='store_true', help='Whether we select the logically correct most probable one during evaluation')
    parser.add_argument('--masked_cases', default='', type=str, help="The value cases we don't want the model to predict so we mask these cases in gold with the mask token")

    # Inference
    parser.add_argument('--model_to_load_path', default='', type=str, help="The path of the model to be loaded. E.g: experiments/exp__20220425_165912/best_model.pt")

    args = parser.parse_args()

    args.data_dir = os.path.join(Config.DATA_PREFIX, args.data_set)
    args.model_output_dir = Config.EXPERIMENT_PREFIX
    args.inference_output_dir = Config.INFERENCE_PREFIX

    print("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        print("argument: {}={}".format(argument, getattr(args, argument)))

    return args
