import json

def fix_notebook_dataloading():
    notebook_path = "kaggle_training.ipynb"
    
    print(f"Loading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # We need to find the cell where 'dataset_path' is defined and ADD the loading logic
    
    loading_logic = [
        "    print(f\"Correctly configured to load: {dataset_path}\")\n",
        "    \n",
        "    # --- INJECTED LOADING LOGIC ---\n",
        "    with open(dataset_path, 'r', encoding='utf-8') as f:\n",
        "        data_json = json.load(f)\n",
        "        \n",
        "    # Convert to Hugging Face Dataset format\n",
        "    # Handle 'val' vs 'validation' key difference just in case\n",
        "    val_key = 'validation' if 'validation' in data_json else 'val'\n",
        "    \n",
        "    raw_datasets = DatasetDict({\n",
        "        \"train\": Dataset.from_list(data_json[\"train\"]),\n",
        "        \"validation\": Dataset.from_list(data_json[val_key]),\n",
        "        \"test\": Dataset.from_list(data_json[\"test\"])\n",
        "    })\n",
        "    print(f\"Data loaded successfully! Structure: {raw_datasets}\")\n",
        "    \n",
        "except FileNotFoundError:\n",
        "    print(f\"WARNING: Dataset not found at {dataset_path}. Please check Kaggle upload.\")\n"
    ]
    
    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Join to checking content
            content = "".join(source)
            
            if "dataset_path =" in content and "Check if file exists" in content:
                print("Found Target Cell. Injecting logic...")
                
                # We need to replace the end of the try block
                # The original code ended with:
                #     print(f"Correctly configured to load: {dataset_path}")
                # except FileNotFoundError: ...
                
                # We will re-write the source code list carefully
                
                # Keep the top part (imports + path definition + try + open check)
                # We essentially replace:
                #    print(f"Correctly configured to load: {dataset_path}")\n
                #    except FileNotFoundError:\n
                #    print(f"WARNING: Dataset not found at {dataset_path}. Please check Kaggle upload.\")\n"
                
                # With our new loading_logic block (which includes the exception block at the end)
                
                # Let's find index where the print happens
                split_idx = -1
                for i, line in enumerate(source):
                    if "Correctly configured to load" in line:
                        split_idx = i
                        break
                
                if split_idx != -1:
                    new_source = source[:split_idx] + loading_logic
                    cell['source'] = new_source
                    found = True
                    break
    
    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Notebook patched successfully!")
    else:
        print("ERROR: Could not find target cell pattern.")

if __name__ == "__main__":
    fix_notebook_dataloading()
