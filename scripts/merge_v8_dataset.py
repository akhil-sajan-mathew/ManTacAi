import json
import random

def merge_v8_dataset():
    # Paths
    base_path = "dataset_augmented/v7_training_data_complete_sanitized.json" 
    patch_path_1 = "dataset_augmented/v8_boring_emergency_1300.json"
    patch_path_2 = "dataset_augmented/v8_persuasion_real_800.json"
    patch_path_3 = "dataset_augmented/v8_gaslighting_real_600.json"
    output_path = "dataset_augmented/v8_training_data_final.json"
    
    print(f"Loading Base: {base_path}...")
    with open(base_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
        
    print(f"Loading Patches...")
    with open(patch_path_1, 'r', encoding='utf-8') as f:
        patch_1 = json.load(f)
    with open(patch_path_2, 'r', encoding='utf-8') as f: 
        patch_2 = json.load(f)
    with open(patch_path_3, 'r', encoding='utf-8') as f: 
        patch_3 = json.load(f)

    # 1. Flatten Base & FILTER OUT BAD DATA
    val_key = "validation" if "validation" in base_data else "val"
    all_base_items = base_data['train'] + base_data.get(val_key, []) + base_data['test']
    
    print(f"Original Base Count: {len(all_base_items)}")
    
    clean_base = []
    removed_context_count = 0
    removed_gaslight_count = 0
    
    for item in all_base_items:
        text = item.get('text', '').strip().lower()
        source = item.get('source', '')
        
        # Filter 1: "Context:" garbage (Persuasion)
        if text.startswith("context:"):
            removed_context_count += 1
            continue
            
        # Filter 2: "gaslighting_train.json" (Internal Monologue)
        # Note: We check if 'source' ends with 'gaslighting_train.json'
        # Or if we want to be safe, we rely on the specific content structure being replaced
        if "gaslighting_train.json" in source:
             removed_gaslight_count += 1
             continue
             
        clean_base.append(item)
            
    print(f"Removed 'Context:' Garbage: {removed_context_count}")
    print(f"Removed 'Gaslight' Monologue: {removed_gaslight_count}")
    print(f"Clean Base Count: {len(clean_base)}")
    
    # 2. Add Patches (Boring + Emergency + Persuasion + Gaslighting)
    combined_items = clean_base + patch_1 + patch_2 + patch_3
    total_count = len(combined_items)
    print(f"Merged Count: {total_count} (+{len(patch_1)} V8_1, +{len(patch_2)} V8_2, +{len(patch_3)} V8_3)")
    
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
        
    print(f"V8 Merge Complete! Saved to {output_path}")

if __name__ == "__main__":
    merge_v8_dataset()
