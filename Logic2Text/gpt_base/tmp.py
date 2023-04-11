#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import os


def main():
    d = {"sha1":[], "text":[]}
    hashed_inferred_texts = []
    with open("./dataset/FIX_original_data/test.json") as f:
        original_json = json.load(f)
        for i, table in enumerate(original_json):
            hashed_inferred_texts.append("{sha1} {text}".format(sha1=table["sha1"], text=table["sent"]))
            d["sha1"].append(table["sha1"])
            d["text"].append(table["sent"])

    out_hashed_csv_path = os.path.join("./output_inference", "hashed_inferred_texts.csv")
    pd.DataFrame(data=d).to_csv(out_hashed_csv_path, index=False)

    out_hashed_path = os.path.join("./output_inference", "hashed_inferred_texts.txt")
    with open(out_hashed_path, "w") as f:
        f.write("\n".join(hashed_inferred_texts))

if __name__ == '__main__':
    main()