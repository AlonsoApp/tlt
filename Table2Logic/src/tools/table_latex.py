import json

from logic2text.utils import build_table

if __name__ == '__main__':
    dataset_path = "./data/Logic2Text/original_data_fix/all_data.json"
    with open(dataset_path) as f:
        data_in = json.load(f)
    hashed_dataset = {}
    for sample in data_in:
        hashed_dataset[sample["sha1"]] = sample
    while True:
        query_sha1 = input("Sha1: ")
        sample = hashed_dataset[query_sha1]
        df = build_table(sample)
        print(df.to_latex(index=False, bold_rows=True))