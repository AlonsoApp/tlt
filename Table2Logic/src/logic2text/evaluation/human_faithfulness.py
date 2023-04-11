import json
from typing import List

from tabulate import tabulate
import random
import hashlib
import re
import numpy as np
import copy
import sys

from collections import Counter

from logic2text.utils import build_table, build_tree

import pandas as pd
import os
import datetime

from sklearn.metrics import cohen_kappa_score

def three_true_false_sha_list(path):
    """
    Get the sha1s of the samples in 3 that contains a True LF
    :return:
    """
    true_list = []
    false_list = []
    with open(os.path.join(path, "3", "dataset", "test.json")) as f:
        three_dataset = json.load(f)
    for sample in three_dataset:
        pd_table = build_table(sample)
        ast_tree = build_tree(sample, pd_table)
        if ast_tree.execute(pd_table):
            true_list.append(sample["sha1"])
        else:
            false_list.append(sample["sha1"])
    return true_list, false_list

def inference_df(path):
    """
    Generates a df with sha1 as a key and three columns text_3, text_4 and text_5
    :param path:
    :return:
    """
    inf_dfs = {}
    for exp in ["3", "4", "5"]:
        df = pd.read_csv(os.path.join(path, exp, "hashed_inferred_texts.csv"))
        df = df.set_index('sha1')
        inf_dfs[exp] = df
    merge_df = inf_dfs["3"].join(inf_dfs["4"], lsuffix='_3', rsuffix='_4')
    merge_df = merge_df.join(inf_dfs["5"])
    merge_df = merge_df.rename(columns={"text": "text_5"})
    merge_df = merge_df.dropna()
    return merge_df

def parse_answer(answer):
    answer = answer.replace(" ", "")
    answer = answer.split(",")
    res_1 = answer[0]
    res_2 = answer[1] if len(answer) > 1 else ''
    return res_1, res_2

def create_eval_folder(path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exp = "eval_{}".format(timestamp)

    out_path = os.path.join(path, exp)
    os.makedirs(out_path, exist_ok=True)

    return out_path

def eval_questionary(inf_df, hashed_dataset, sha1_list, output_path, eval_type):
    evaluation_csv_path = os.path.join(output_path, "eval_{}.csv".format(eval_type))
    evaluation_df = pd.DataFrame({'sha1': [],
                                  'text_3': [], 'text_4': [], 'text_5': [],
                                  'eval_3': [], 'eval_4': [], 'eval_5': [],
                                  'error_type_3': [], 'error_type_4': [], 'error_type_5': []})
    max_iter = 100
    for i, sha1 in enumerate(sha1_list[22:max_iter]):
        infs = inf_df.loc[[sha1]]
        pd_table = build_table(hashed_dataset[sha1])
        print("Progress: {}/{}".format(i+22, max_iter))
        print("sha1: {}".format(sha1))
        print("Topic: {}".format(hashed_dataset[sha1]["topic"]))
        print(tabulate(pd_table, headers='keys', tablefmt='psql'))
        evaluation_dic = {"sha1": sha1}
        for exp in ["3", "4", "5"]:
            print("{}: {}".format(exp, infs["text_" + exp][0]))
            evaluation_dic["text_" + exp] = infs["text_" + exp][0]
        for exp in ["3", "4", "5"]:
            answer = input("Eval {}: ".format(exp))
            evaluation, error_type = parse_answer(answer)
            evaluation_dic["eval_" + exp] = evaluation
            evaluation_dic["error_type_" + exp] = error_type
        evaluation_df = evaluation_df.append(evaluation_dic, ignore_index=True)
        evaluation_df.to_csv(evaluation_csv_path)

def analyze_results_individual():
    path = "./evaluation/eval_20220601_121554/eval_true.csv"
    df = pd.read_csv(path)
    total_len = len(df)
    for exp in ["3", "4", "5"]:
        print("Exp {}:".format(exp))
        for res in ["t", "f", "n", "h"]:
            res_val = len(df[df["eval_" + exp] == res])
            print("{}: {} ({}%)".format(res, res_val, int((res_val/total_len)*100)))
        for res in ["1", "2", "3", "4"]:
            res_val = len(df[df["error_type_" + exp].str.contains(res, na=False)])
            print("{}: {}".format(res, res_val))

def build_df(path, exps):
    regex_id = r"^id: (\b[0-9a-f]{5,40}\b) - (\b[0-9a-f]{5,40}\b)"
    regex_answer = r"^Answer \[t\/f\/n]: ([t|f|n])"
    regex_sent = r"Frase: (.+)"
    result_docs = [x for x in os.listdir(path)]
    ref_exp = {}
    df_dic = {'sha1': [], 'exp': [], 'variation': [], 'sent': [], 'answer': []}
    for exp in exps:
        sha1 = hashlib.sha1(bytes("text_" + str(exp), "utf-8")).hexdigest()
        ref_exp[sha1] = exp
    for result_doc in result_docs:
        variation = result_doc.replace('.txt', '')[-1]
        with open(os.path.join(path,result_doc),  encoding="utf8", errors='ignore') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            sha1 = ""
            exp = ""
            sent = ""
            check = False
            for line in lines:
                id_search = re.search(regex_id, line, re.IGNORECASE)
                if id_search:
                    if check:
                        raise exec("Two sha1 detected in a row.")
                    sha1 = id_search.group(1)
                    exp = ref_exp[id_search.group(2)]
                    check = True
                sent_search = re.search(regex_sent, line, re.IGNORECASE)
                if sent_search:
                    sent = sent_search.group(1)
                answer_search = re.search(regex_answer, line, re.IGNORECASE)
                if answer_search:
                    answer = answer_search.group(1)
                    df_dic["sha1"].append(sha1)
                    df_dic["exp"].append(exp)
                    df_dic["variation"].append(variation)
                    df_dic["sent"].append(sent)
                    df_dic["answer"].append(answer)
                    check = False


    df = pd.DataFrame(data=df_dic)
    #df.to_csv(os.path.join(path, "results.csv"))
    return df


def fleiss_kappa(df: pd.DataFrame, options: List[str], question_id_col: str, answer_col: str):
    """
    Given a df for a document calculates Fleiss' Kappa for inter evaluator agreement
    source: https://en.wikipedia.org/wiki/Fleiss%27_kappa
    < 0	Poor agreement
    0.01 – 0.20	Slight agreement
    0.21 – 0.40	Fair agreement
    0.41 – 0.60	Moderate agreement
    0.61 – 0.80	Substantial agreement
    0.81 – 1.00	Almost perfect agreement
    :param df: dataframe
    :param options: answer options e.g.: ['t','f','n']
    :param question_id_col: name of the column that identifies each question e.g: sha1
    :param answer_col: name of the column that has the answer e.g: answer
    :return:
    """
    question_ids = df[question_id_col].unique()
    matrix = [[0 for x in range(len(options))] for y in range(len(question_ids))]
    for i, q_id in enumerate(question_ids):
        answers = df[df[question_id_col] == q_id][answer_col]
        for n, answer in answers.items():
            j = options.index(answer)
            matrix[i][j] += 1
    #print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in matrix]))
    total = sum(sum(matrix,[]))
    p = []
    for opt in range(len(options)):
        t = sum([q[opt] for q in matrix])
        p.append(t/total)
    P = []
    for q in matrix:
        n = -sum(q)
        t = (sum([np.power(val, 2) for val in q])-n)/(n*(n-1))
        P.append(t)

    P_h = sum(P)/len(question_ids)
    Pe_h = sum([np.power(val, 2) for val in p])

    k = (P_h - Pe_h)/(1 - Pe_h)
    return k, matrix

def print_matrix(matrix):
    printable_m = copy.deepcopy(matrix)
    for i, question in enumerate(printable_m):
        question.insert(0, "q{}".format(i + 1))
    printable_m.insert(0, ['', 't', 'f', 'n'])

    print('\n'.join([''.join(['{}	'.format(item) for item in row]) for row in printable_m]))

def cohens_kappa(df):
    results = {'a': [], 'b': []}
    sha1_list = df['sha1'].unique()
    for sha1 in sha1_list:
        df_question = df[df['sha1'] == sha1]
        for key in results.keys():
            results[key].append(df_question[df_question['variation']==key]['answer'].values[0])
    eval1 = results['a']
    eval2 = results['b']
    #from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    #import matplotlib.pyplot as plt
    #cm = confusion_matrix(eval1, eval2)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot()
    #plt.show()
    k = cohen_kappa_score(eval1, eval2)
    return k

def analyze_results_doc():
    path = "./evaluation/eval_20220605_174730/results"
    exps = ["3", "4", "5"]
    df = build_df(path, exps)
    for exp in exps:
        df_exp = df[df['exp'] == exp]
        print("Exp {}:".format(exp))
        sha1_list = df_exp['sha1'].unique()
        # Count vote result of t, f, n per question
        results = {'t': 0, 'f': 0, 'n': 0, '-': 0}
        for sha1 in sha1_list:
            df_question = df_exp[df_exp['sha1'] == sha1]
            win, total = Counter(df_question['answer'].tolist()).most_common()[0]
            if total > 1:
                results[win] += 1
            else:
                results['-'] += 1
        full_disagreement = results.pop('-')
        total = sum([x for x in results.values()])
        for res, res_val in results.items():
            print("{}: {} ({}%)".format(res, res_val, round((res_val/total)*100, 2)))
        print("-: {} ".format(full_disagreement))

        # Count total t, f and n
        #for res in ["t", "f", "n"]:
        #    res_val = len(df_exp[df_exp["answer"] == res])
        #    print("{}: {} ({}%)".format(res, res_val, int((res_val/len(df_exp))*100)))

        # Agreement
        f_k, matrix = fleiss_kappa(df_exp, options=['t', 'f', 'n'], question_id_col='sha1', answer_col='answer')
        c_k = cohens_kappa(df_exp)
        #cohen_kappa_score(df_exp)
        #print("Table of values")
        #print_matrix(matrix)
        print("Fleiss: {:.4f}".format(f_k))
        print("Cohens: {:.4f}".format(c_k))


def play():
    path = "./evaluation/eval_20220601_121554/eval_true.csv"
    df = pd.read_csv(path)
    exp = "5"
    res = "h"
    sha1 = df[df["eval_" + exp].str.contains(res, na=False)]
    #sha1 = df[df["error_type_" + exp].str.contains(res, na=False)]
    print(tabulate(sha1, headers='keys', tablefmt='psql'))

def individual_evaluation():
    dataset_path = "./data/Logic2Text/original_data_fix/test.json"
    inference_path = "./inferences/output_logic2text/t2l"
    evaluation_path = create_eval_folder("./evaluation")
    with open(dataset_path) as f:
        data_in = json.load(f)
    hashed_dataset = {}
    for sample in data_in:
        hashed_dataset[sample["sha1"]] = sample
    inf_df = inference_df(inference_path)
    three_true_sha1_list, three_false_sha1_list = three_true_false_sha_list(inference_path)
    print("################# Evaluating TRUE LFs #################")
    eval_questionary(inf_df, hashed_dataset, three_true_sha1_list, evaluation_path, "true")
    print("################# Evaluating FALSE LFs #################")
    eval_questionary(inf_df, hashed_dataset, three_false_sha1_list, evaluation_path, "false")

def doc_planning(sha1_list):
    exps = ["3", "4", "5"]
    num_samples_to_eval = 90
    num_evaluators = 18
    num_duplicates = 2
    num_exp = len(exps)
    num_unique_docs = int(num_evaluators / num_duplicates)
    question_per_doc = num_samples_to_eval * num_exp / num_unique_docs
    random.shuffle(sha1_list)
    docs = [[] for _ in range(num_unique_docs)]
    for sha1 in sha1_list[:num_samples_to_eval]:
        sha1_exps = [(sha1, "text_" + str(exp)) for exp in exps]
        unfilled_docs = list(filter(lambda d: len(d) < question_per_doc, docs))
        random.shuffle(unfilled_docs)
        for i, sha1_exp in enumerate(sha1_exps):
            index = i - len(unfilled_docs) * int(i / len(unfilled_docs))
            unfilled_docs[index].append(sha1_exp)
    for doc in docs:
        assert len(doc) == question_per_doc
    return docs


def generate_docs(docs, evaluation_path, inf_df, hashed_dataset):
    for i, doc in enumerate(docs):
        with open(os.path.join(evaluation_path, "doc_{}.txt".format(i+1)), 'a') as doc_file:
            doc_file.write("Instrucciones\n")
            doc_file.write("A continuación verás {n} frases acompañadas de una tabla y su título (de la tabla). "
                           "Las frases dicen algo de la tabla y tendrás que decir si lo que dice "
                           "es cierto (t) o falso (f) en base a la información presente en la tabla y el título. \n"
                           "Si lo que se dice en la frase no está respaldado por ningún dato de la tabla o del título, la "
                           "frase se considerará falsa. \n"
                           "Si la frase menciona información que pueda ser deducida de la tabla como resultado de de "
                           "sumas, restas, medias, etc... la frase se considerará correcta.\n"
                           "Puede que algunas frases estén mal escritas o que digan obviedades como 'in the 1975 "
                           "formula one season, the race on march 1st was the only one to take place on march 1st'. En "
                           "estos casos, si la frase se puede entender y lo que dice sigue siendo cierto, se considerarán "
                           "ciertas (t).\n"
                           "Hay casos extremos en los que las frases pueden no tener ningún sentido y sea "
                           "imposible valorar si lo que dicen es cierto o falso. En estos casos la respuesta debe ser "
                           "'n' de nonsense.\n"
                           "Escribe tu respuesta t, f o n en 'Answer [t/f/n]: ' por ejemplo para una frase correcta -> "
                           "Answer [t/f/n]: t\n"
                           "Cuando termines, guarda el archivo de texto y evíamelo de vuelta a alonsoapp@gmail.com o "
                           "por el canal que más fácil te resulte (Telegram, etc...)\n"
                           "Si tienes alguna duda durante la evaluación preguntame sin problema :)\n"
                           "Te agradezco mucho la ayuda ^^\n\n"
                           "PARA VER BIEN LAS TABLAS: Si usas Sublime Text (cosa que recomiendo), vete a View y "
                           "desactiva Word Wrap. Así, las tablas que no quepan en horizontal no pasarán a la siguiente "
                           "línea (haciéndolas muy difíciles de leer).\n"
                           "Si usas otro editor y quieres saber cómo desactivarlo, busca cómo desabilitar el 'line wrap' en google o pregúntame a mí."
                           "\n\n\n".format(n=len(doc)))
            for i, (sha1, sent) in enumerate(doc):
                pd_table = build_table(hashed_dataset[sha1])
                doc_file.write("# {}/{}\n".format(i+1, len(doc)))
                doc_file.write("id: {} - {}\n".format(sha1, hashlib.sha1(bytes(sent, "utf-8")).hexdigest()))
                doc_file.write("\n")
                doc_file.write("Título: {}\n".format(hashed_dataset[sha1]["topic"]))
                doc_file.write(tabulate(pd_table, headers='keys', tablefmt='psql'))
                doc_file.write("\n")
                doc_file.write("Frase: "+inf_df.loc[[sha1]][sent][0])
                doc_file.write("\n")
                doc_file.write("Answer [t/f/n]: \n")
                doc_file.write("\n")
                doc_file.write("\n")

def build_doc():
    dataset_path = "./data/Logic2Text/original_data_fix/test.json"
    inference_path = "./inferences/output_logic2text/t2l"
    evaluation_path = create_eval_folder("./evaluation")
    with open(dataset_path) as f:
        data_in = json.load(f)
    hashed_dataset = {}
    for sample in data_in:
        hashed_dataset[sample["sha1"]] = sample
    inf_df = inference_df(inference_path)

    docs = doc_planning(list(hashed_dataset.keys()))

    generate_docs(docs, evaluation_path, inf_df, hashed_dataset)

def get_sample(data_json, sha1):
    for sample in data_json:
        if sample['sha1'] == sha1:
            return sample

def analyze_false(exp):
    """
    Shows topic, table and the logical form tha was fed to the model (exp) for each sentence that was marked as false
    :param exp:
    :return:
    """
    path = "./evaluation/eval_20220605_174730/results"
    dataset_path = "./inferences/output_logic2text/t2l/{}/dataset/test.json".format(exp)
    with open(dataset_path) as f:
        data_in = json.load(f)

    df = build_df(path, ["3", "4", "5"])

    df_exp = df[df['exp'] == exp]
    sha1_list = df_exp['sha1'].unique()
    # Count vote result of t, f, n per question
    results = {'t': [], 'f': []}
    for sha1 in sha1_list:
        df_question = df_exp[df_exp['sha1'] == sha1]
        win, total = Counter(df_question['answer'].tolist()).most_common()[0]
        if total > 1 and win == 'f':
            sample = get_sample(data_in, sha1)
            pd_table = build_table(sample)
            ast_tree = build_tree(sample, pd_table)
            print("sha1: {}".format(sha1))
            print("Caption: {}".format(sample["topic"]))
            print(tabulate(pd_table, headers='keys', tablefmt='psql'))
            ast_tree.print_graph()
            print("Sent: {}".format(df_question['sent'].iloc[0]))
            print("Sent gold: {}".format(sample["sent"]))
            print()
            print("Eval2: ")
            print()
            print("------------------------------------------------------------------------")
            #result = input("Result: ")
            #results[result].append(sha1)
    #print("Results:")
    #print("True ({}):".format(len(results['t'])))
    #for x in results['t']:
    #    print(x)
    #print("False ({}):".format(len(results['f'])))
    #for x in results['f']:
    #    print(x)

def analyze_true(exp="3"):
    """
    Shows topic, table, the gen LF and the gold LF for cases where genLF != goldLF but exp 3 == faithful
    :return:
    """
    path = "./evaluation/eval_20220605_174730/results"
    dataset_path = "./inferences/output_logic2text/t2l/{}/dataset/test.json".format(exp)
    with open(dataset_path) as f:
        data_in = json.load(f)
    with open("./inferences/output_logic2text/t2l/{}/dataset/test.json".format(4)) as f:
        data_gold = json.load(f)
    with open("./inferences/output_logic2text/t2l/{}/dataset/test.json".format(0)) as f:
        data_human = json.load(f)

    df = build_df(path, ["3", "4", "5"])

    df_exp = df[df['exp'] == exp]
    df_gold = df[df['exp'] == "4"]
    sha1_list = df_exp['sha1'].unique()
    # Count vote result of t, f, n per question
    results = {'t': [], 'f': []}
    for sha1 in sha1_list:
        df_question = df_exp[df_exp['sha1'] == sha1]
        df_question_gold = df_gold[df_gold['sha1'] == sha1]
        win, total = Counter(df_question['answer'].tolist()).most_common()[0]
        if total > 1 and win == 't':
            sample_gen = get_sample(data_in, sha1)
            sample_gold = get_sample(data_gold, sha1)
            sample_human = get_sample(data_gold, sha1)
            pd_table = build_table(sample_gen)
            ast_tree_gen = build_tree(sample_gen, pd_table)
            ast_tree_gold = build_tree(sample_gold, pd_table)
            gen_action_str = " ".join([str(x) for x in ast_tree_gen.to_action_list()])
            gold_action_str = " ".join([str(x) for x in ast_tree_gold.to_action_list()])
            if gen_action_str != gold_action_str:
                # gen sentence is faithful but its LF != gold LF
                print("sha1: {}".format(sha1))
                print("Caption: {}".format(sample_gen["topic"]))
                print(tabulate(pd_table, headers='keys', tablefmt='psql'))
                print("gen LF:")
                ast_tree_gen.print_graph()
                print("gold LF:")
                ast_tree_gold.print_graph()
                print("Sent gen: {}".format(df_question['sent'].iloc[0]))
                win_gold, total_gold = Counter(df_question_gold['answer'].tolist()).most_common()[0]
                gold_grade = win_gold if total_gold > 1 else "50%"
                print("Sent gold ({}): {}".format(gold_grade, df_question_gold["sent"].iloc[0]))
                print("Sent human: {}".format(sample_human["sent"]))
                print()
                print("------------------------------------------------------------------------")

if __name__ == '__main__':
    #individual_evaluation()
    #analyze_results_individual()
    #play()
    #build_doc()
    #analyze_results_doc()
    #analyze_false("4")
    if sys.argv[1] == 'build_doc':
        build_doc()
    elif sys.argv[1] == 'analyze_results_doc':
        analyze_results_doc()
    elif sys.argv[1] == 'analyze_false':
        analyze_false(sys.argv[2])
    elif sys.argv[1] == 'analyze_true':
        analyze_true()




