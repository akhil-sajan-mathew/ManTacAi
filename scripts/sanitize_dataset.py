import json
import random

def sanitize_dataset():
    input_path = "dataset_augmented/v7_training_data_complete.json"
    output_path = "dataset_augmented/v7_training_data_complete_sanitized.json"
    
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 1. Flatten all data
    val_key = "validation" if "validation" in data else "val"
    all_items = data['train'] + data.get(val_key, []) + data['test']
    print(f"Original Count: {len(all_items)}")
    
    # 2. De-Duplicate (The Leakage Fix)
    seen_texts = set()
    clean_items = []
    duplicates_removed = 0
    
    for item in all_items:
        # Normalize text for comparison
        text_signature = item['text'].strip().lower()
        
        if text_signature not in seen_texts:
            seen_texts.add(text_signature)
            clean_items.append(item)
        else:
            duplicates_removed += 1
            
    print(f"Duplicates Removed: {duplicates_removed}")
    print(f"Clean Count: {len(clean_items)}")
    
    # 3. Re-Shuffle & Re-Split (80/10/10)
    # This ensures the clean data is distributed evenly
    random.shuffle(clean_items)
    
    total = len(clean_items)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_set = clean_items[:train_end]
    val_set = clean_items[train_end:val_end]
    test_set = clean_items[val_end:]
    
    print(f"New Splits:\n  Train: {len(train_set)}\n  Val: {len(val_set)}\n  Test: {len(test_set)}")
    
    # 4. Save
    final_data = {
        "train": train_set,
        "validation": val_set,
        "test": test_set
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)
        
    print(f"Sanitized dataset saved to {output_path}")

if __name__ == "__main__":
    sanitize_dataset()
