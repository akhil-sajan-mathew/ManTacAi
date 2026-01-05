import json
import random
import os

def merge_datasets():
    # Paths
    v5_path = os.path.join("dataset_augmented", "v5_training_data_final.json")
    v6_path = os.path.join("dataset_augmented", "v6_normalcy_1250.json")
    output_path = os.path.join("dataset_augmented", "v6_training_data_final.json")
    
    print(f"Loading V5 from {v5_path}...")
    with open(v5_path, 'r', encoding='utf-8') as f:
        v5_data = json.load(f)
        
    print(f"Loading V6 from {v6_path}...")
    with open(v6_path, 'r', encoding='utf-8') as f:
        v6_data = json.load(f)
        
    # Shuffle V6 to ensure random distribution across splits
    random.shuffle(v6_data)
    
    total_v6 = len(v6_data)
    print(f"Total V6 items: {total_v6}")
    
    # Calculate Split Indices (80% Train, 10% Val, 10% Test)
    train_end = int(total_v6 * 0.8)
    val_end = train_end + int(total_v6 * 0.1)
    
    v6_train = v6_data[:train_end]
    v6_val = v6_data[train_end:val_end]
    v6_test = v6_data[val_end:]
    
    print(f"Splitting V6: Train={len(v6_train)}, Val={len(v6_val)}, Test={len(v6_test)}")
    
    # Merge into V5 structure
    # V5 structure is expected to be {"train": [], "validation": [], "test": []}
    # We need to handle case if keys are missing or named differently (e.g. "val" vs "validation")
    
    # Check keys
    keys = v5_data.keys()
    print(f"V5 Keys: {list(keys)}")
    
    # Robust key mapping
    train_key = "train"
    val_key = "validation" if "validation" in keys else "val"
    test_key = "test"
    
    if train_key not in v5_data:
        print("ERROR: Could not find 'train' key in V5 dataset.")
        return

    # Append Data
    v5_data[train_key].extend(v6_train)
    
    if val_key in v5_data:
        v5_data[val_key].extend(v6_val)
    else:
        print("Warning: Validation key not found, creating it.")
        v5_data["validation"] = v6_val
        
    if test_key in v5_data:
        v5_data[test_key].extend(v6_test)
    else:
        print("Warning: Test key not found, creating it.")
        v5_data["test"] = v6_test
        
    # Check valid JSON strictness (ensure no trailing commas etc by default dump)
    
    print(f"Saving merged dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(v5_data, f, indent=4)
        
    print("Merge Complete!")
    print(f"Final Counts -> Train: {len(v5_data[train_key])}, Val: {len(v5_data.get(val_key, []))}, Test: {len(v5_data.get(test_key, []))}")

if __name__ == "__main__":
    merge_datasets()
