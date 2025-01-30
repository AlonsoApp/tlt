# TlT: Automatic Logical Forms improve fidelity in Table-to-Text generation

Authors: _Iñigo Alonso_, _Eneko Agirre_

Code submitted to the journal _Expert Systems with Applications_.

This folder contains two projects: **Table2Logic** and **Logic2Text**. Both projects contain `README.md` files explaining how they work. We advise reviewers to first read [`Table2Logic/README.md`](https://github.com/AlonsoApp/tlt/tree/main/Table2Logic) as it contains all the steps needed to replicate this work's results.

# Table2Logic

## Environment
We use **Python 3.8**<br>
All package requirements are listed in `./requirements.txt`
```
pip install -r ./requirements.txt
```
**Working directory:**<br> 
We launch all scripts from the root directory of this project.

## Data
1. Get the dataset files in the data supplementary material or clone the [original Logic2Text GitHub repository](https://github.com/czyssrs/Logic2Text)
2. Copy all .json files from `/Logic2Text/dataset` folder from the cloned repo to `./data/Logic2Text/original_data` 
folder in this project. Data folder should look like this `./data/Logic2Text/original_data/JSON_FILES`
3. Run:
```
python ./src/logic2text/utils/fix_all.py
```
This will generate original_data_fix folder containing the dataset that will later be used in this project. Other 
generated folders can be deleted as they are intermediate checkpoints of the modification process, they will not be used
again. Read `./data/Logic2Text/README.md` to know more about the fixes performed to the original Logic2Text dataset.

## Content Selection Values
In the paper, the manually extracted Content Selection value categories are named differently to make them easy to 
understand. Internally these categories are called cases.<br>
These are the naming equivalences between the paper and the project:
* TAB = case1a + case1b
* INF = case2
* AUX = case3

## Training a new Table2Logic model
To train a new Table2Logic model with the same configuration as TLT mentioned in the paper run:
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case1a;case1b;case2;case3" --masked_cases "" --cuda --include_oov_token --rejection_sampling
```
Parameters:
* --batch_size: Batch size used for training
* --beam_size: Beam size used on inference time
* --value_cases_in_extra: Content Selection value categories manually extracted form the gold logical forms and fed to the model
* --masked_cases: Content Selection value categories that will be masked out as OOV in the reference logical forms
* --cuda: use cuda GPUs to perform the training
* --include_oov_token:  Allows Value Pointer Network to produce OOV tokens. All paper models have this.
* --rejection_sampling: Whether o not use False Candidate Rejection (internally called rejection_sampling) in inference time.

To see a full list of parameters got to `./src/config.py`. All published models use different combinations of the just
mentioned parameters. The rest of the parameters shown in this file are kept to their default values.

IMPORTANT: any Content Selection Value categories (called 'cases' in this project) not represented in 
--value_cases_in_extra must be added to --masked_cases so the values of these categories are masked out as OOV in the 
reference logical forms. 

## Models from the paper
Unfortunately the trained models shown in the paper exceed the maximum file size allowed as supplementary material. 
However, each model can be easily retrained running the following scripts:

**No CS (TLT no CS)**
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "" --masked_cases "case1b;case2;case3" --cuda --include_oov_token
```
**TAB**
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case1a;case1b" --masked_cases "case2;case3" --cuda --include_oov_token
```
**INF**
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case2" --masked_cases "case1b;case3" --cuda --include_oov_token
```
**AUX**
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case3" --masked_cases "case1b;case2;" --cuda --include_oov_token
```
**TAB, INF**
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case1a;case1b;case2" --masked_cases "case3" --cuda --include_oov_token
```
**TAB, AUX**
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case1a;case1b;case3" --masked_cases "case2" --cuda --include_oov_token
```
**TAB, INF, AUX**
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case1a;case1b;case2;case3" --masked_cases "" --cuda --include_oov_token
```
**TAB, INF, AUX + FCR (TlT)**
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case1a;case1b;case2;case3" --masked_cases "" --cuda --include_oov_token --rejection_sampling
```

Trained models can be found in `./experiments` folder. 

## Inference
Load a trained Table2Logic model and generate new logical forms. This will use test.json samples from dataset and 
generate a new logical form for each sample using the loaded Table2Logic model. This new test.json can be found in 
`./inferences/` folder.
```
./src/main_inference.py --beam_size 2048 --value_cases_in_extra "case1a;case1b;case2;case3" --masked_cases "" --model_to_load_path "experiments/exp_1a-1b-2-3_RS/best_model.pt" --cuda --include_oov_token --rejection_sampling
```
In addition to the previously explained parameters, inference mode offers a new one: 
* --model_to_load_path: Path the trained model checkpoint is located

## Full workflow
1. Get the dataset ready to be used
```
python ./src/logic2text/utils/fix_all.py
```
2. Train a model with the desired configuration. (In this example we will replicate TLT)
```
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case1a;case1b;case2;case3" --masked_cases "" --cuda --include_oov_token --rejection_sampling
```
3. Use a trained Table2Logic model to generate a copy of the test dataset replacing its logical forms with the ones 
generated by that model. Go to `./experiments/` and look for the trained Table2Logic model of your choice. It can be the 
model you just trained. You can also use an already trained one like TLT `./experiments/exp_1a-1b-2-3_RS/best_model.pt`.
```
python ./src/main_inference.py --beam_size 2048 --value_cases_in_extra "case1a;case1b;case2;case3" --model_to_load_path "experiments/exp_1a-1b-2-3_RS/best_model.pt" --cuda --include_oov_token --rejection_sampling
```
4. Generate the test.text file needed for Logic2Text. Look for the folder where the just generated text.json is, it 
should be at `./inferences/`.
```
python ./src/logic2text/utils/generate_text_files.py ./inferences/exp__20220922_205533
```
5. Go to Logic2Text project and copy the generated `test.json` and `test.text` files into `dataset/original_data` and run:
```
python ./preprocess.py ../dataset/ ../gpt_models/
```
6. Now make inference with the Logic2Text model to produce the sentences related to the generated logical forms. This uses 
the logic2Text model saved in that path. To see how to train your own Logic2Text model go to the README.md in Logic2Text project.  
```
inference.py --mode test --saved_model_path ../output_gpt/03_fix_l2t_20220519180620/saved_model/loads/14/
```
7. Generated texts will be saved at `output_inference` folder in the Logic2Text project.

The texts generated by this workflow for the different model configurations shown in section 'Model configurations'
of the paper can be found in `./inferences/output_logic2text/t2l`. Each number corresponds to a model configuration in
the paper:
* 1 = T2T no CS
* 2 = TLT no CS
* 3 = TLT
* 4 = TLT gold
* 5 = T2T

## Fidelity evaluation
The folder `./evaluation/eval_20220605_174730` contains the questionnaires used for the 'Fidelity evaluation' section in 
the paper. Each doc_n.txt was given to three different evaluators. The results of these questionnaires can be found in 
`./evaluation/eval_20220605_174730/results/`. The variants a and b represent the answers of different evaluators for the 
same questionary.

To see how questionnaires where made, go to `./src/logic2text/evaluation/human_faithfulness.py` and see the `build_doc()` 
function.

To create a new set of questionnaires run:
```
python ./src/logic2text/evaluation/human_faithfulness.py build_doc
```
Generated questionnaires will appear in `./evaluation/` folder.

To analyze the results of the evaluation from the paper and see the results shown in the section 'Fidelity evaluation' 
run:
```
python ./src/logic2text/evaluation/human_faithfulness.py analyze_results_doc
```

Model configurations follow the same naming convention shown in the previous section:
* 1 = T2T no CS
* 2 = TlT no CS
* 3 = TLT
* 4 = TLT gold
* 5 = T2T

## Automatic evaluation
To get the results featured in section 'Automatic evaluation' of the paper run:
```
python ./src/logic2text/evaluation/automatic_eval.py
```
This script uses all inferences from folder `./inferences/output_logic2text/t2l`, which contains the inferred texts from
all model configurations shown in the paper with the addition of:
* 0 = Contains the human written reference texts. They will be used as a reference to compute each of the metrics.

As shown at the end of the _Full workflow_ section of this README, these folders contain the texts generated by the 
Logic2Text model variation that corresponds to its configuration (see Figure 3 in the paper to see model 
configurations). They also contain the dataset used to generate such texts. Each dataset follows the same format as the 
original but with different logical forms depending on the model configuration that produced them. Some are gold logical
forms and some others are the logical forms generated by the Table2Logic model variation associated to that 
configuration. Each of these texts is paired to the sha1 used as unique identifier of each sample in the dataset. 

## Qualitative analysis
### Error cases at the table-to-logic stage
To get the qualitative analysis results for the error cases at the table-to-logic stage run:
```
python ./src/logic2text/evaluation/qualitative_analysis.py
```

### Error cases at the logic-to-text stage
To get all the samples over which we conducted the qualitative analysis of error cases at the logic-to-text stage run:
```
python ./src/logic2text/evaluation/human_faithfulness.py analyze_false
```
Read _Fidelity evaluation_ section of this README to understand where this samples come from.

### Can an incorrect LF produce a faithful description?
To get all the samples where the automatic LFs from TLT resulted in faithful sentences in the manual evaluation while 
being different from their gold LF references run:
```
python ./src/logic2text/evaluation/human_faithfulness.py analyze_true
```
Read _Fidelity evaluation_ section of this README to understand where this samples come from.

## Acknowledgements
This project makes use of the code from [Valuenet](https://github.com/brunnurs/valuenet) and 
[Logic2Text](https://github.com/czyssrs/Logic2Text).

## Reference
If you find this project useful, please cite it using the following format

```
@article{ALONSO2024121869,
  title = {Automatic Logical Forms improve fidelity in Table-to-Text generation},
  journal = {Expert Systems with Applications},
  volume = {238},
  pages = {121869},
  year = {2024},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2023.121869},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417423023710},
  author = {Iñigo Alonso and Eneko Agirre},
  keywords = {Natural Language Generation, Table-to-Text, Deep learning, Logical forms, Faithfulness, Hallucinations}
}
```

