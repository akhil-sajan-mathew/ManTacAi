import json
import random

def merge_neutral_patch():
    # Paths
    base_path = "dataset_augmented/v7_training_data_sanitized.json" # The clean one
    patch_path = "dataset_augmented/v7_neutral_patch_500.json" # The new neutral items
    output_path = "dataset_augmented/v7_training_data_complete.json" # The Final Final
    
    print(f"Loading Base: {base_path}...")
    with open(base_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
        
    print(f"Loading Patch: {patch_path}...")
    with open(patch_path, 'r', encoding='utf-8') as f:
        patch_data = json.load(f)
        
    # 1. Flatten Base
    val_key = "validation" if "validation" in base_data else "val"
    all_items = base_data['train'] + base_data[val_key] + base_data['test']
    print(f"Base Count: {len(all_items)}")
    
    # 2. Add Patch
    combined_items = all_items + patch_data
    total_count = len(combined_items)
    print(f"Merged Count: {total_count} (+{len(patch_data)})")
    
    # 3. Shuffle & Re-Split (80/10/10)
    random.shuffle(combined_items)
    
    train_end = int(total_count * 0.8)
    val_end = int(total_count * 0.9)
    
    train_set = combined_items[:train_end]
    val_set = combined_items[train_end:val_end]
    test_set = combined_items[val_end:]
    
    print(f"New Splits:\n  Train: {len(train_set)}\n  Val: {len(val_set)}\n  Test: {len(test_set)}")
    
    # 4. Save
    final_data = {
        "train": train_set,
        "validation": val_set,
        "test": test_set
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)
        
    print(f"Complete! Saved to {output_path}")

if __name__ == "__main__":
    merge_neutral_patch()
