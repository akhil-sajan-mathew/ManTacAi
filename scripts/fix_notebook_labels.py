import json

def fix_notebook_labels():
    notebook_path = "kaggle_training.ipynb"
    
    print(f"Loading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Logic to inject
    # We need to map string labels ("manipulation_tactic") to integers using label2id
    new_preprocess_logic = [
        "def preprocess_function(examples):\n",
        "    tokenized = tokenizer(examples[\"text\"], truncation=True, max_length=512)\n",
        "    # Map labels if they exist\n",
        "    if \"manipulation_tactic\" in examples:\n",
        "        tokenized[\"label\"] = [label2id[label] for label in examples[\"manipulation_tactic\"]]\n",
        "    return tokenized\n"
    ]
    
    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            content = "".join(source)
            if "def preprocess_function(examples):" in content:
                print("Found Preprocess Function. Updating logic...")
                
                # We replace the old function definition line and the return line
                # The old source looks like:
                #    def preprocess_function(examples):
                #        return tokenizer(examples["text"], truncation=True, max_length=512)
                
                # We can just match the start of the function and replace the whole block
                # finding start index
                start_idx = -1
                end_idx = -1
                
                for i, line in enumerate(source):
                    if "def preprocess_function(examples):" in line:
                        start_idx = i
                    if "return tokenizer" in line and start_idx != -1:
                        end_idx = i
                        break
                
                if start_idx != -1 and end_idx != -1:
                    # Construct new source: Everything before start + new logic + Everything after end
                    new_source = source[:start_idx] + new_preprocess_logic + source[end_idx+1:]
                    cell['source'] = new_source
                    found = True
                    break

    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Notebook patched successfully!")
    else:
        print("ERROR: Could not find preprocess_function to patch.")

if __name__ == "__main__":
    fix_notebook_labels()
