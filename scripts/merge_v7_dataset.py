import json
import random
import os

def merge_datasets():
    # Paths
    v6_path = "dataset_augmented/v6_training_data_final.json"
    v7_path = "dataset_augmented/v7_nuance_400.json"
    output_path = "dataset_augmented/v7_training_data_final.json"
    
    # 1. Load V6 (Base)
    print(f"Loading V6 from {v6_path}...")
    try:
        with open(v6_path, 'r', encoding='utf-8') as f:
            v6_data = json.load(f)
    except FileNotFoundError:
        print("Error: V6 dataset not found.")
        return

    # 2. Load V7 (New Nuance Data)
    print(f"Loading V7 from {v7_path}...")
    try:
        with open(v7_path, 'r', encoding='utf-8') as f:
            v7_data = json.load(f)
    except FileNotFoundError:
        print("Error: V7 dataset not found.")
        return

    # 3. Shuffle V7
    random.shuffle(v7_data)
    total_v7 = len(v7_data)
    
    # 4. Split V7 (80/10/10)
    train_end = int(total_v7 * 0.8)
    val_end = int(total_v7 * 0.9)
    
    v7_train = v7_data[:train_end]
    v7_val = v7_data[train_end:val_end]
    v7_test = v7_data[val_end:]
    
    print(f"V7 Splits - Train: {len(v7_train)}, Val: {len(v7_val)}, Test: {len(v7_test)}")

    # 5. Merge into V6 Structure
    # Handle key naming (some versions use 'val' vs 'validation')
    val_key = "validation" if "validation" in v6_data else "val"
    
    final_data = {
        "train": v6_data["train"] + v7_train,
        val_key: v6_data[val_key] + v7_val,
        "test": v6_data["test"] + v7_test
    }
    
    # 6. Stats
    total_train = len(final_data["train"])
    total_val = len(final_data[val_key])
    total_test = len(final_data["test"])
    
    print(f"FINAL V7 DATASET COUNTS:\nTrain: {total_train}\nVal: {total_val}\nTest: {total_test}")

    # 7. Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)
        
    print(f"Merge Complete! Saved to {output_path}")

if __name__ == "__main__":
    merge_datasets()
