import json

def update_notebook_v8():
    notebook_path = "kaggle_training.ipynb"
    
    print(f"Loading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # --- 1. Fix Markdown Instructions ---
    # Find the first markdown cell
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            content = "".join(cell['source'])
            if "Instructions" in content:
                print("Updating Instructions...")
                new_source = [
                    "# Manipulation Tactic Detector - Training Notebook (V8)\n",
                    "\n",
                    "This notebook fine-tunes the `j-hartmann/emotion-english-distilroberta-base` model on the ManTacAi dataset (V8 Platinum).\n",
                    "\n",
                    "## Instructions\n",
                    "1. Upload the `dataset_augmented/v8_training_data_final.json` file to your Kaggle environment as a dataset named `v8-final-balanced`.\n",
                    "2. Run all cells to train the model.\n",
                    "3. Download the `manipulation_tactic_detector_model.zip` file at the end."
                ]
                cell['source'] = new_source
                break
                
    # --- 2. Fix Label Map & Dataset Path ---
    # We need to construct the Python code exactly as it should appear
    # The Correct 18-Class Map
    new_code_source = [
        "# Label Mapping (V8 - 18 Classes)\n",
        "id2label = {\n",
        "    0: \"threatening_intimidation\",\n",
        "    1: \"gaslighting\", \n",
        "    2: \"guilt_tripping\",\n",
        "    3: \"deflection\",\n",
        "    4: \"stonewalling\",\n",
        "    5: \"belittling_ridicule\",\n",
        "    6: \"love_bombing\",\n",
        "    7: \"ethical_persuasion\", \n",
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
        "# Updated path for V8 Final (Boring + Emergency + Real Persuasion + Real Gaslighting)\n",
        "dataset_path = \"/kaggle/input/v8-final-balanced/v8_training_data_final.json\"\n",
        "\n",
        "try:\n",
        "    with open(dataset_path, 'r', encoding='utf-8') as f:\n",
        "    # Check if file exists to give user feedback immediately\n",
        "        pass\n",
        "    print(f\"Correctly configured to load: {dataset_path}\")\n",
        "except FileNotFoundError:\n",
        "    print(f\"WARNING: Dataset not found at {dataset_path}. Please check Kaggle upload.\")\n"
    ]
    
    # Locate the cell that defines id2label
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            content = "".join(cell['source'])
            if "id2label = {" in content:
                print("Updating Label Map & Dataset Path...")
                cell['source'] = new_code_source
                break
                
    # --- 3. Save ---
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        
    print(f"Notebook updated successfully!")

if __name__ == "__main__":
    update_notebook_v8()
