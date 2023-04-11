## Add 1
We added the field _sha1_ to every sample of each dataset, so it can act as a unique identifier of that sample. The sha1 
is composed by the sha1 of the url + action of each sample. We use this identifier later on for analytics.

## Fix 1
The LFs in string format provided in the original dataset have an ambiguity on the functions hop, eq, not_eq, all_eq, 
all_not_eq, most_eq, most_not_eq, filter_eq, filter_not_eq in which the functionality of those functions is divided in two
implementations; one for their normal version and another for when the eq comparison must be performed at string level.
Logic2Text chooses which functionality to run given the name of the function provided in the LFs json representation 
(provided also with every sample). This breaks the translation parity between the LF provided in the "logic_str" and 
"logic" objects. To fix this, we consider these disambiguation function names as different functions in the grammar. 
Therefore, we generate the logic_str by building the string out of the "logic" object provided with each sample 
(which contains the correct disambiguated names of each function)

## Fix 2
The original LF grammar presented in Logic2Text has an inconsistency where
hop (which returns a unique value of a column given a row) gets a View instead of a Row. This is
grammatically incorrect because what happens if it gets a View with multiple rows? Which value should hop return? This
inconsistency is present in the 25% of the original dataset samples. To fix this we changed the
'hop' keyword of all 'hop View C' rules to 'hop_first'. Making it coherent through all the grammar.

This dataset is the exact same dataset as the original one provided in Logic2Text except for the hop keyword is changed
to hop_first when it is used in a View and not a Row.

The only logic str where this is applied is the logic_str of the root of each sample. The logic object is still
the same.


## Fix 3
There are four types of values in the Logic Forms:
* case 1a: (26.64%) Values that reference table values. These values should be present in the table literally
* case 1b: (45.52%) Values that reference table values but don't have to be strictly equal to table values. The str used in these values is chosen by the LF author
* case 2: (20.79%) Values that are the outcome of an arithmetic operation
* case 3: (7.05%) Values chosen by the author of the LF. Are often used as comparative values such as bigger than or less than

The problem is that 330 of the 10753 LFs contain at least one case 1 value which can't be found literally in the table.
These LFs validate correctly because these values are used, for example as a filter in filter_eq operations. The
implementation of these operators use sub string "contains" operators instead of hard equality of strings. However, as
the grammar definition clearly states that table values must be equal to the query value, these 330 cases have been
removed from the dataset.
* all: 10509 out of 10753 original values
Removed: 244 samples
* test: 1070 out of 1092 original values
Removed: 22 samples
* train: 8374 out of 8566 original values
Removed: 192 samples
* valid: 1065 out of 1095 original values
Removed: 30 samples

## Fix 4
Considering a limitation of input tokens in Bert. Our baseline decoder can not be feed with an input longer than 512 
tokens. Thus, we removed all the samples that would exceed this max input size after all the pre-processing. 
* all: 9522 out of 10509 original values
Removed 987 samples
* test: 972 out of 1070 original values
Removed 98 samples
* train: 7559 out of 8374 original values
Removed 815 samples
* valid: 991 out of 1065 original values
Removed 74 samples

## (new) With case 1a and case 1b
Fix 1: Ambiguous grammar
100%|██████████████████████████████████| 10753/10753 [00:00<00:00, 38822.17it/s]
100%|████████████████████████████████████| 1092/1092 [00:00<00:00, 16927.33it/s]
100%|████████████████████████████████████| 8566/8566 [00:00<00:00, 32906.41it/s]
100%|████████████████████████████████████| 1095/1095 [00:00<00:00, 41779.36it/s]
Fix 2: hop to hop_first
100%|████████████████████████████████████| 10753/10753 [00:22<00:00, 471.11it/s]
100%|██████████████████████████████████████| 1092/1092 [00:01<00:00, 551.65it/s]
100%|██████████████████████████████████████| 8566/8566 [00:16<00:00, 518.87it/s]
100%|██████████████████████████████████████| 1095/1095 [00:02<00:00, 483.77it/s]
Fix 3: Case 1 values not found in table
100%|████████████████████████████████████| 10753/10753 [00:13<00:00, 782.87it/s]
Saving 10753 out of 10753 original values
Removed: 0 samples
100%|██████████████████████████████████████| 1092/1092 [00:01<00:00, 797.30it/s]
Saving 1092 out of 1092 original values
Removed: 0 samples
100%|██████████████████████████████████████| 8566/8566 [00:09<00:00, 884.34it/s]
Saving 8566 out of 8566 original values
Removed: 0 samples
100%|██████████████████████████████████████| 1095/1095 [00:01<00:00, 786.31it/s]
Saving 1095 out of 1095 original values
Removed: 0 samples
Fix 4: Max tokens
100%|████████████████████████████████████| 10753/10753 [00:49<00:00, 215.35it/s]
Saving 9706 out of 10753 original values
Removed 1047 samples
100%|██████████████████████████████████████| 1092/1092 [00:04<00:00, 234.58it/s]
Saving 988 out of 1092 original values
Removed 104 samples
100%|██████████████████████████████████████| 8566/8566 [00:35<00:00, 242.00it/s]
Saving 7702 out of 8566 original values
Removed 864 samples
100%|██████████████████████████████████████| 1095/1095 [00:04<00:00, 244.86it/s]
Saving 1016 out of 1095 original values
Removed 79 samples