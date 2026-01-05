import json
import re

def clean_persuasion_data():
    input_path = "dataset_augmented/v8_training_data_final.json"
    output_path = "dataset_augmented/v8_training_data_cleaned.json"
    
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Flatten
    val_key = "validation" if "validation" in data else "val"
    all_items = data['train'] + data.get(val_key, []) + data['test']
    
    cleaned_count = 0
    dropped_count = 0
    clean_items = []
    
    # Regex patterns for the "Separator"
    # Matches "Phrase:" or "Dialogue:" or "Says:" case insensitive
    separator_pattern = re.compile(r'(?:phrase|dialogue|says|response)\s*:', re.IGNORECASE)

    for item in all_items:
        text = item['text']
        
        # Only target the polluted rows
        if text.strip().lower().startswith("context:"):
            # Attempt to find separator
            match = separator_pattern.search(text)
            
            if match:
                # Keep everything AFTER the match
                start_index = match.end()
                clean_text = text[start_index:].strip()
                
                # Remove wrapping quotes if present
                clean_text = clean_text.strip('"').strip("'")
                
                item['text'] = clean_text
                # Tag it so we know it was modified
                item['category'] = "cleaned_persuasion" 
                clean_items.append(item)
                cleaned_count += 1
            else:
                # FAIL SAFETY: If we can't find the split, we drop it.
                # Better to lose data than train on "Context: ..."
                dropped_count += 1
        else:
            # Pass through healthy data
            clean_items.append(item)
            
    print(f"Cleaning Complete.")
    print(f"  - Repaired: {cleaned_count} rows")
    print(f"  - Dropped:  {dropped_count} rows (Could not extract dialogue)")
    print(f"  - Total Remaining: {len(clean_items)}")
    
    # Re-structure (No need to re-shuffle/split really, but good practice to keep structure)
    # Actually, let's keep the size constant-ish.
    
    # Simple split strategy to finish
    import random
    random.shuffle(clean_items)
    
    split_1 = int(len(clean_items) * 0.8)
    split_2 = int(len(clean_items) * 0.9)
    
    final_data = {
        "train": clean_items[:split_1],
        "validation": clean_items[split_1:split_2],
        "test": clean_items[split_2:]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)
        
    print(f"Saved cleaned dataset to {output_path}")

if __name__ == "__main__":
    clean_persuasion_data()
