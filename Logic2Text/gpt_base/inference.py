#!/usr/bin/env python
# -*- coding: utf-8 -*-

from SeqUnit import *
from DataLoader import DataLoader, Preprocessor
import model as model_gpt
from tqdm import tqdm
import encoder
import hashlib
import pandas as pd

from utils import *


# paths and mode
tf.app.flags.DEFINE_string("prog_name",'gpt_base','program name')

tf.app.flags.DEFINE_string("root_path", "../dataset/", "full path of data folder")
tf.app.flags.DEFINE_string("gpt_model_path",'../gpt_models','path of gpt2 model')
tf.app.flags.DEFINE_string("gpt_model_name",'117M','gpt2 model')
tf.app.flags.DEFINE_string("output_path", "../output_inference", "full path of saved output")
tf.app.flags.DEFINE_string("model_save_name", "tmp", "full path of saved output")


tf.app.flags.DEFINE_string("mode",'test','train or test')

tf.app.flags.DEFINE_string("model_dir",'','specify model dir name')

tf.app.flags.DEFINE_boolean("use_table", True,'input table or not') # use table

# for resume training
tf.app.flags.DEFINE_string("resume_path",'','saved model path for use in resume mode')
tf.app.flags.DEFINE_string("resume_model_path",'','saved model path for use in resume model')

# for testing
tf.app.flags.DEFINE_string("saved_model_path",'','saved model path for use in test mode')

tf.app.flags.DEFINE_string("decoding",'beam','greedy or beam for decoding')
tf.app.flags.DEFINE_integer("beam_size", 2,'beam search for decoding')

# for table only
tf.app.flags.DEFINE_integer("max_input_len", 500, 'max input len')
tf.app.flags.DEFINE_integer("max_text_len", 50, 'max text len')
tf.app.flags.DEFINE_integer("max_table_len", 100, 'max table len')

# architecture choices
tf.app.flags.DEFINE_boolean("use_coverage", False,'use coverage or not')
tf.app.flags.DEFINE_float("coverage_penalty", 0.02,'coverage loss penalty')
tf.app.flags.DEFINE_boolean("use_copy_gate", True,'use copy gate or not')
tf.app.flags.DEFINE_float("copy_gate_penalty", 0.7, 'copy gate loss penalty')

# data options
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')
tf.app.flags.DEFINE_integer("source_vocab", 50257,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 50257,'vocabulary size')

# model hyperparams
tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 768, "Size of embedding.")

# training
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size of train set.")
tf.app.flags.DEFINE_integer("batch_size_test", 1, "Batch size of test set.")
tf.app.flags.DEFINE_integer("batch_update", 32, "apply gradients after steps")
tf.app.flags.DEFINE_integer("epoch", 5000, "Number of training epoch.")
tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')

# logging
tf.app.flags.DEFINE_integer("report", 50,'report valid results after some steps')
tf.app.flags.DEFINE_integer("report_loss", 10,'report loss results after some steps')

FLAGS = tf.app.flags.FLAGS

# create output paths
saved_model_path = os.path.join(FLAGS.output_path, FLAGS.saved_model_path)
os.makedirs(FLAGS.output_path, exist_ok=True)
log_file = os.path.join(FLAGS.output_path, 'log.txt')


# create data paths
root_path = FLAGS.root_path
processed_data_dir = os.path.join(root_path, "processed_data")
original_data_dir = os.path.join(root_path, "original_data")

# bpe vocab
last_best = 0.0
enc = encoder.get_encoder("117M", FLAGS.gpt_model_path)
eos = 50256 #TODO move to settings
empty = 2 #TODO move to settings

def inference(sess, preprocessed_data, model, ksave_dir, mode):
    datasets = {"all_data":preprocessed_data.all_data_set, "train":preprocessed_data.train_set, "test":preprocessed_data.test_set, "valid":preprocessed_data.dev_set}
    data_iterator = DataLoader(FLAGS.use_table, datasets[mode],
                               bpe_enc=FLAGS.gpt_model_path, batch_size=FLAGS.batch_size_test, shuffle=False, eos=eos,
                               empty=empty, man_input_len=FLAGS.max_input_len, man_text_len=FLAGS.max_text_len,
                               man_table_len=FLAGS.max_table_len)

    os.makedirs(ksave_dir, exist_ok=True)
    inferred_texts = []
    k = 0
    for x in tqdm(data_iterator):
        ### beam search batch size = 1
        beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all = model.generate_beam(x, sess)
        cand = np.array(cand_seqs_all).tolist()
        cand_score = np.array(cand_probs_all).tolist()

        ind = cand_score.index(max(cand_score))

        res_cand = cand[ind]

        if eos in res_cand:
            res_cand = res_cand[:res_cand.index(eos)] if res_cand[0] != eos else [eos]

        real_sum = enc.decode(res_cand[1:])

        real_sum = real_sum.replace("\n", " ").strip()

        if not check_res(real_sum):
            real_sum = "empty ."

        inferred_texts.append(real_sum)

        k += 1

    out_real_path = os.path.join(ksave_dir,  "inferred_texts.txt")
    with open(out_real_path, "w") as f:
        f.write("\n".join(inferred_texts))
    return inferred_texts


def main():
    mode = "test"
    gpt_model_name = os.path.join(FLAGS.gpt_model_path, FLAGS.gpt_model_name)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        hparams = model_gpt.default_hparams()
        with open(os.path.join(gpt_model_name, 'hparams.json')) as f:   
            hparams.override_from_dict(json.load(f))

        preprocessed_data = Preprocessor(processed_data_dir, FLAGS.limits, eos, empty, bpe_enc=FLAGS.gpt_model_path, 
                                            max_text_len=FLAGS.max_text_len, max_input_len=FLAGS.max_input_len)

        model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size,
                        emb_size=FLAGS.emb_size, source_vocab=FLAGS.source_vocab, 
                        target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                        learning_rate=FLAGS.learning_rate, use_coverage = FLAGS.use_coverage,
                        coverage_penalty=FLAGS.coverage_penalty,
                        copy_gate_penalty=FLAGS.copy_gate_penalty, use_copy_gate=FLAGS.use_copy_gate,
                        gpt_hparams=hparams, decoding=FLAGS.decoding, beam_size=FLAGS.beam_size, empty_token=empty, stop_token=eos, 
                        max_length=FLAGS.max_text_len)

        print (saved_model_path)
        model.load(saved_model_path, sess)
        inferred_texts = inference(sess, preprocessed_data, model, FLAGS.output_path, mode)

    # Last, lets create a file with the inferenced texts paired with their corresponding table represented as de sha1(url)
    original_data = os.path.join(original_data_dir, "{}.json".format(mode))
    d = {"sha1":[], "text":[]}
    hashed_inferred_texts = []
    with open(original_data) as f:
        original_json = json.load(f)
        assert len(original_json)==len(inferred_texts)
        for i, table in enumerate(original_json):
            hashed_inferred_texts.append("{sha1} {text}".format(sha1=table["sha1"], text=inferred_texts[i]))
            d["sha1"].append(table["sha1"])
            d["text"].append(inferred_texts[i])

    out_hashed_csv_path = os.path.join(FLAGS.output_path, "hashed_inferred_texts.csv")
    pd.DataFrame(data=d).to_csv(out_hashed_csv_path, index=False)

    out_hashed_path = os.path.join(FLAGS.output_path, "hashed_inferred_texts.txt")
    with open(out_hashed_path, "w") as f:
        f.write("\n".join(hashed_inferred_texts))

if __name__ == '__main__':
    main()