import json
import os

def update_notebook():
    nb_path = "kaggle_training.ipynb"
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    print("Loaded notebook. Finding config cell...")
    
    # 1. Update Configuration Cell (Index 5 based on inspection)
    # Search for cell containing "id2label = {"
    config_cell_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and "id2label = {" in "".join(cell["source"]):
            config_cell_idx = i
            break
            
    if config_cell_idx == -1:
        print("ERROR: Could not find configuration cell.")
        return

    print(f"Found config cell at index {config_cell_idx}. Updating...")
    
    # New Source Code for Cell
    new_source = [
        "# Label Mapping (V6 - 18 Classes)\n",
        "id2label = {\n",
        "    0: \"threatening_intimidation\",\n",
        "    1: \"gaslighting\", \n",
        "    2: \"guilt_tripping\",\n",
        "    3: \"deflection\",\n",
        "    4: \"stonewalling\",\n",
        "    5: \"belittling_ridicule\",\n",
        "    6: \"love_bombing\",\n",
        "    7: \"threatening_intimidation\", \n",
        "    8: \"passive_aggression\",\n",
        "    9: \"appeal_to_emotion\",\n",
        "    10: \"whataboutism\",\n",
        "    11: \"neutral_conversation\",\n",
        "    12: \"coercive_control\",\n",
        "    13: \"benign_venting\",\n",
        "    14: \"healthy_conflict\",\n",
        "    15: \"benign_affection\",\n",
        "    16: \"neutral_logistics\",\n",
        "    17: \"urgent_emergency\"\n",
        "}\n",
        "label2id = {v: k for k, v in id2label.items()}\n",
        "\n",
        "# Load Dataset\n",
        "# Updated path for V8 Final (Boring + Emergency Boosted)\n",
        "dataset_path = \"/kaggle/input/v8-final-balanced/v8_training_data_final.json\"\n",
        "\n",
        "try:\n",
        "    with open(dataset_path, 'r', encoding='utf-8') as f:\n",
        "        data = json.load(f)\n",
        "    print(f\"Successfully loaded dataset. Keys: {list(data.keys())}\")\n",
        "except FileNotFoundError:\n",
        "    print(f\"ERROR: File '{dataset_path}' not found. Please ensure the dataset is added to the notebook.\")\n",
        "    # Fallback for local testing if needed, or stop\n",
        "    raise\n",
        "\n",
        "def create_dataset_object(data_list):\n",
        "    # Filter out any items with unknown labels\n",
        "    valid_items = [\n",
        "        item for item in data_list \n",
        "        if item[\"manipulation_tactic\"] in label2id\n",
        "    ]\n",
        "    if len(valid_items) < len(data_list):\n",
        "        print(f\"Warning: Filtered out {len(data_list) - len(valid_items)} items with unknown labels.\")\n",
        "        \n",
        "    return Dataset.from_list([\n",
        "        {\n",
        "            \"text\": item[\"text\"],\n",
        "            \"label\": label2id[item[\"manipulation_tactic\"]]\n",
        "        } \n",
        "        for item in valid_items\n",
        "    ])\n",
        "\n",
        "# Handle different key names for validation split\n",
        "val_key = \"validation\" if \"validation\" in data else \"val\"\n",
        "\n",
        "if val_key in data and \"test\" in data:\n",
        "    raw_datasets = DatasetDict({\n",
        "        \"train\": create_dataset_object(data[\"train\"]),\n",
        "        \"validation\": create_dataset_object(data[val_key]),\n",
        "        \"test\": create_dataset_object(data[\"test\"])\n",
        "    })\n",
        "else:\n",
        "    print(\"Splitting 'train' set as validation/test sets were not found.\")\n",
        "    full_ds = create_dataset_object(data[\"train\"])\n",
        "    splits = full_ds.train_test_split(test_size=0.2, seed=42)\n",
        "    test_val = splits[\"test\"].train_test_split(test_size=0.5, seed=42)\n",
        "    raw_datasets = DatasetDict({\n",
        "        \"train\": splits[\"train\"],\n",
        "        \"validation\": test_val[\"train\"],\n",
        "        \"test\": test_val[\"test\"]\n",
        "    })\n",
        "\n",
        "print(raw_datasets)"
    ]
    
    nb["cells"][config_cell_idx]["source"] = new_source
    
    # Save
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        
    print("Notebook updated successfully!")

if __name__ == "__main__":
    update_notebook()
