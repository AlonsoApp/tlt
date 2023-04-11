# Workflow

Use a trained Table2Logic model to generate a copy of the test dataset but with LFs generated by that model at model_to_load_path
```
python ./src/main_inference.py --beam_size 8 --value_cases_in_extra "case1a;case1b;case2;case3" --model_to_load_path "experiments/exp__20220504_124717/best_model.pt" --cuda --include_oov_token --rejection_sampling
```

Generate test.text file needed for Logic2Text
```
python ./src/logic2text/utils/generate_text_files.py ./inferences/NAME_OF_EXP_FOLDER
```

Go to Logic2Text project and copy the generated `test.json` and `test.text` files into `dataset/original_data` and run:
```
python ./preprocess.py ../dataset/ ../gpt_models/
```

Now make inference with Logic2Text model to produce the sentences related to the generated logical forms. This uses 
the logic2Text model saved in that path. Texts will be saved at `output_inference` folder
```
inference.py --mode test --saved_model_path ../output_gpt/tmp_20220519180620/saved_model/loads/14/
```