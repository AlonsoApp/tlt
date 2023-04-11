import os
import pandas as pd
import datasets
from evaluate import load
from logic2text.evaluation.utils.bart_score import BARTScorer
from tabulate import tabulate
from numpy import mean

def calc_bleu(df):
	refs = [[x] for x in df['text_ref'].tolist()]
	hyp = df['text_hyp'].tolist()
	metric = datasets.load_metric('sacrebleu')
	metric.add_batch(predictions=hyp, references=refs)
	score = metric.compute()
	return {'bleu-4': score['score']}


def calc_rouge(df):
	refs = df['text_ref'].tolist()
	hyp = df['text_hyp'].tolist()
	metric = datasets.load_metric('rouge')
	metric.add_batch(predictions=hyp, references=refs)
	score = metric.compute()
	result = {'rouge1': score['rouge1'].mid.fmeasure,
			  'rouge2': score['rouge2'].mid.fmeasure,
			  'rougeL': score['rougeL'].mid.fmeasure}
	return result

def calc_bertscore(df):
	"""
	src: https://huggingface.co/spaces/evaluate-metric/bertscore
	:param df:
	:return:
	"""
	refs = df['text_ref'].tolist()
	hyp = df['text_hyp'].tolist()
	bertscore = load("bertscore")
	results = bertscore.compute(predictions=hyp, references=refs, lang="en")
	result = {'bertscore-precision': mean(results['precision']),
			  'bertscore-recall': mean(results['recall']),
			  'bertscore-f1': mean(results['f1'])}
	return result

def calc_bartscore(df):
	"""
	src: https://github.com/neulab/BARTScore
	:param df:
	:return:
	"""
	refs = df['text_ref'].tolist()
	hyp = df['text_hyp'].tolist()
	# To use the CNNDM version BARTScore
	bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
	score = bart_scorer.score(hyp, refs)  # generation scores from the first list of texts to the second list of texts.
	return {'bartscore': mean(score)}

def main(reference, path):
	experiments = [x for x in next(os.walk(path))[1]]
	experiments.remove(reference)
	reference_df = pd.read_csv(os.path.join(path, reference, "hashed_inferred_texts.csv"))
	reference_df = reference_df.set_index('sha1')
	all_score_dic = []
	for exp in experiments:
		hyp_df = pd.read_csv(os.path.join(path, exp, "hashed_inferred_texts.csv"))
		hyp_df = hyp_df.set_index('sha1')
		merge_df = reference_df.join(hyp_df, lsuffix='_ref', rsuffix='_hyp')
		merge_df = merge_df.dropna()
		bleu_score = calc_bleu(merge_df)
		rouge_score = calc_rouge(merge_df)
		bertscore = calc_bertscore(merge_df)
		bartscore = calc_bartscore(merge_df)
		scores = {'Exp': exp, **bleu_score, **rouge_score, **bertscore, **bartscore}
		all_score_dic.append(scores)
	df = pd.DataFrame(all_score_dic)
	df = df.sort_values('Exp')
	print(tabulate(df, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
	main("0", "./inferences/output_logic2text/t2l")