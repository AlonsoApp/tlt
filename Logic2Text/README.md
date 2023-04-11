# Logic2Text

## Data
Uncompress `supplementary_materia_data.tgz` from the submitted Supplementary material

[Download](https://www.dropbox.com/sh/99awpjnj2lh4e17/AACCz_XU_FhkinSId0_nz1-qa?dl=0) the original GPT-2 pre-trained model form the dropbox url provided in the offical Logic2Text GitHub repository. The copy its contents to `./gpt_models/`.

This folder should look like this `./gpt_models/117M`

## Requirements
python 3.6
tensorflow 1.12

## Data
1. Copy all .json files from `/dataset` folder in the supplementary material to `./dataset/original_data` folder in this project. Data folder should look like this `./dataset/original_data/JSON_FILES` 

or

1. Clone the [original Logic2Text GitHub repository](https://github.com/czyssrs/Logic2Text)
2. Copy all .json files from `/Logic2Text/dataset` folder from the cloned repo to `./dataset/original_data` folder in this project. Data folder should look like this `./dataset/original_data/JSON_FILES`

then

1. Run:
```
python ./gpt_base/generate_text_files.py ./dataset/original_data
```
2. Pre-process the data:
```
python ./gpt_base/preprocess.py ../dataset/ ../gpt_models/
```

## Models
Unfortunately the trained models shown in the paper exceed the maximum file size allowed as supplementary material. 
However, in this section we will explain how the can be replicated:
### Logic2Text
```
cd ./gpt_base/
python Main.py --mode train
```
### Logic2Text for T2T no CS
1. Go to `./gpt_base/DataLoader.py`
2. Look for the comment # comment if test no logic
3. Comment the line below
4. Run:
```
cd ./gpt_base/
python Main.py --mode train
```
### Logic2Text for T2T
1. Go to `./gpt_base/DataLoader.py`
2. Look for the comment # comment if test no logic
3. Comment the line below
4. In the same file look for the comment # uncomment this to use extra_values (Content Selection values)
5. Uncomment the line below
6. Run:
```
cd ./gpt_base/
python Main.py --mode train
```

Trained models can be found in `./output_gpt`.

For more info about Logic2Text project go to [Logic2Text GitHub](https://github.com/czyssrs/Logic2Text)
