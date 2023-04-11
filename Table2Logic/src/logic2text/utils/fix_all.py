from pathlib import Path
import glob
import shutil
import os
import fix_ambiguous_grammar, fix_dataset_grammar, fix_dataset_values, fix_max_tokens, add_sha1, generate_text_files

def create_original_data_fix(last_process_path):
	original_data_fix_path = "./data/Logic2Text/original_data_fix/"
	Path(original_data_fix_path).mkdir(parents=True, exist_ok=True)
	# Remove any files within the folder
	files = glob.glob(original_data_fix_path+"*")
	for f in files:
		os.remove(f)
	# Copy files from last path to original_fix
	file_names = os.listdir(last_process_path)
	for file_name in file_names:
		shutil.copy(os.path.join(last_process_path, file_name), original_data_fix_path)
	return original_data_fix_path


if __name__ == '__main__':
	dataset_path = "./data/Logic2Text/original_data/"
	print("Generating text files...")
	dataset_path = generate_text_files.run(dataset_path)
	print("Add 1: Add sha1")
	dataset_path = add_sha1.run(dataset_path)
	print("Fix 1: Ambiguous grammar")
	dataset_path = fix_ambiguous_grammar.run(dataset_path)
	print("Fix 2: hop to hop_first")
	dataset_path = fix_dataset_grammar.run(dataset_path)
	print("Fix 3: Case 1 values not found in table")
	dataset_path = fix_dataset_values.run(dataset_path)
	print("Fix 4: Max tokens")
	dataset_path = fix_max_tokens.run(dataset_path)
	print("Creating original_data_fix")
	dataset_path = create_original_data_fix(dataset_path)
	print("Generating text files...")
	generate_text_files.run(dataset_path)