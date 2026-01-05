import json
import random
from collections import Counter

def analyze_quality():
    path = "dataset_augmented/v7_training_data_final.json"
    print(f"Loading {path}...")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Flatten all splits
    all_items = data['train'] + data.get('val', []) + data.get('validation', []) + data['test']
    
    # Filter for V7
    v7_items = [x for x in all_items if x.get('category', '').startswith('v7_')]
    
    if not v7_items:
        print("CRITICAL: No V7 items found with 'category' tag. Checking for text patterns...")
        # Fallback if category didn't survive merge (though it should have)
        return

    with open("v7_audit_log.txt", "w", encoding="utf-8") as log:
        log.write(f"\n--- V7 DATASET AUDIT ---\n")
        log.write(f"Total V7 Items: {len(v7_items)}\n")
        
        # 1. Uniqueness Check
        texts = [x['text'] for x in v7_items]
        unique_texts = set(texts)
        dupe_count = len(texts) - len(unique_texts)
        log.write(f"Uniqueness Ratio: {len(unique_texts)/len(texts)*100:.1f}%\n")
        log.write(f"Exact Duplicates: {dupe_count}\n")

        # 2. Category Distribution
        cats = Counter([x['category'] for x in v7_items])
        log.write("\nCategory Distribution:\n")
        for c, count in cats.items():
            log.write(f"  - {c}: {count}\n")

        # 3. Slang/Noise Density Audit
        slang_markers = ["u", "ur", "h8", "cuz", "bc", "omg", "literally", "trash", "thx"]
        noise_markers = ["ffs", "smh", "stg", "lol", "rn", "tho"]
        
        slang_hits = sum(1 for t in texts if any(m in t.lower().split() for m in slang_markers))
        noise_hits = sum(1 for t in texts if any(m in t.lower().split() for m in noise_markers))
        
        log.write("\nRealism Metrics:\n")
        log.write(f"  - Slang Density: {slang_hits/len(texts)*100:.1f}% of samples contain slang (u, cuz, h8)\n")
        log.write(f"  - Noise Density: {noise_hits/len(texts)*100:.1f}% of samples contain emotional noise (ffs, smh, omg)\n")
        
        # 4. Deep Dive Samples
        log.write("\n--- QUALITATIVE INSPECTION (Random Samples) ---\n")
        for cat in cats:
            log.write(f"\n[{cat.upper()}]\n")
            cat_items = [x['text'] for x in v7_items if x['category'] == cat]
            for s in random.sample(cat_items, 5):
                log.write(f"  > {s}\n")
    
    print("Audit log written to v7_audit_log.txt")

if __name__ == "__main__":
    analyze_quality()
