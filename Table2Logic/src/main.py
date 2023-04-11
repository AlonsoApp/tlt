import json
import os

import torch
from pytictoc import TicToc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import read_arguments_train, write_config_to_file
from data_loader import get_data_loader
from logic2text import dataset_utils
from model.model import IRNet
from optimizer import build_optimizer_encoder
from training import train
from evaluation import evaluate
from utils import setup_device, set_seed_everywhere, save_model, create_experiment_folder

from intermediate_representation import lf_grammar


if __name__ == '__main__':
    args = read_arguments_train()

    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
    print("Run experiment '{}'".format(experiment_name))

    write_config_to_file(args, output_path)

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    train_data = dataset_utils.load_dataset(args.data_dir, "train.json", use_small=args.toy)
    val_data = dataset_utils.load_dataset(args.data_dir, "valid.json", use_small=args.toy)
    train_loader = get_data_loader(train_data, args.batch_size, True)
    val_loader = get_data_loader(val_data, args.batch_size, False)

    grammar = lf_grammar.Grammar()
    model = IRNet(args, device, grammar)
    # model.load_state_dict(torch.load("experiments/exp__20220425_165912/best_model.pt"))
    model.to(device)

    # torch.autograd.set_detect_anomaly(True)

    num_train_steps = len(train_loader) * args.num_epochs
    optimizer, scheduler = build_optimizer_encoder(model,
                                                   num_train_steps,
                                                   args.lr_transformer, args.lr_connection, args.lr_base,
                                                   args.scheduler_gamma)

    train_writer = SummaryWriter(output_path+'/logs/train')
    valid_writer = SummaryWriter(output_path+'/logs/valid')
    global_step = 0
    best_acc = 0.0

    # Train
    print("Start training with {} epochs".format(args.num_epochs))
    t = TicToc()
    for epoch in tqdm(range(int(args.num_epochs))):
        sketch_loss_weight = 1 if epoch < args.loss_epoch_threshold else args.sketch_loss_weight

        t.tic()
        global_step = train(global_step,
                            train_writer,
                            train_loader,
                            model,
                            optimizer,
                            args,
                            sketch_loss_weight=sketch_loss_weight)

        train_time = t.tocvalue()

        tqdm.write("Training of epoch {0} finished after {1:.2f} seconds. Evaluate now on the dev-set".format(epoch,
                                                                                                              train_time))

        with torch.no_grad():
            train_sketch_acc, train_acc, _ = evaluate(model, train_loader, args)
            sketch_acc, acc, predictions = evaluate(model, val_loader, args)

        with open(os.path.join(output_path, 'predictions_action_list.json'), 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)

        eval_results_string = "Epoch: {}    Sketch-Accuracy: {}     Accuracy: {}".format(epoch + 1, sketch_acc, acc)
        tqdm.write(eval_results_string)

        if acc > best_acc:
            save_model(model, os.path.join(output_path))
            tqdm.write(
                "Accuracy of this epoch ({}) is higher than the so far best accuracy ({}). Save model.".format(acc,
                                                                                                               best_acc))
            best_acc = acc

        with open(os.path.join(output_path, "eval_results.log"), "a+", encoding='utf-8') as writer:
            writer.write(eval_results_string + "\n")

        # wandb.log({"Sketch-accuracy": sketch_acc, "accuracy": acc}, step=epoch + 1)

        train_writer.add_scalar("sketch-accuracy", train_sketch_acc, epoch + 1)
        train_writer.add_scalar("accuracy", train_acc, epoch + 1)

        valid_writer.add_scalar("sketch-accuracy", sketch_acc, epoch + 1)
        valid_writer.add_scalar("accuracy", acc, epoch + 1)

        scheduler.step()  # Update learning rate schedule

    train_writer.flush()
    train_writer.close()

    valid_writer.flush()
    valid_writer.close()