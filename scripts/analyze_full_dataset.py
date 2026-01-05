import json
from collections import Counter

def analyze_full_dataset():
    path = "dataset_augmented/v7_training_data_complete_sanitized.json"
    
    with open("final_dataset_audit_log_v2.txt", "w", encoding="utf-8") as log:
        log.write(f"--- FINAL V7 COMPLETE SANITIZED AUDIT ---\n")
        log.write(f"Source: {path}\n\n")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            log.write(f"CRITICAL ERROR: Could not load dataset. {e}\n")
            return

        # 1. SPLIT SIZES
        val_key = "validation" if "validation" in data else "val"
        train = data["train"]
        val = data.get(val_key, [])
        test = data["test"]
        
        total_items = len(train) + len(val) + len(test)
        
        log.write(f"TOTAL SIZE: {total_items}\n")
        log.write(f"  - Train: {len(train)} ({len(train)/total_items*100:.1f}%)\n")
        log.write(f"  - Val:   {len(val)} ({len(val)/total_items*100:.1f}%)\n")
        log.write(f"  - Test:  {len(test)} ({len(test)/total_items*100:.1f}%)\n\n")

        # 2. CLASS BALANCE (The Critical Check)
        all_items = train + val + test
        labels = [x['manipulation_tactic'] for x in all_items]
        label_counts = Counter(labels)
        
        log.write("--- CLASS BALANCE (Global) ---\n")
        # Sort by count desc
        sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
        for label, count in sorted_labels:
            log.write(f"  {label:<30} : {count}\n")
            
        # Check for domination
        max_count = sorted_labels[0][1]
        min_count = sorted_labels[-1][1]
        log.write(f"\n  > Imbalance Ratio (Max/Min): {max_count/min_count:.2f}x\n")

        # 3. LEAKAGE CHECK (Train vs Test)
        train_texts = set(t['text'].strip().lower() for t in train)
        test_texts = set(t['text'].strip().lower() for t in test)
        
        leaks = train_texts.intersection(test_texts)
        log.write(f"\n--- DATA LEAKAGE CHECK ---\n")
        log.write(f"  Exact Text Matches (Train vs Test): {len(leaks)}\n")
        if leaks:
            log.write("  WARNING: Test data is contaminated with training examples!\n")
        else:
            log.write("  PASS: No leakage detected.\n")

        # 4. SLOP CHECK (Global Repetition)
        all_texts = [x['text'] for x in all_items]
        unique_texts = set(all_texts)
        uniqueness = len(unique_texts) / len(all_texts)
        
        log.write(f"\n--- GLOBAL TEXT QUALITY ---\n")
        log.write(f"  Unique Sentences: {len(unique_texts)}/{len(all_texts)} ({uniqueness*100:.1f}%)\n")
        
        # Slang Check
        slang_markers = ["u", "ur", "h8", "cuz", "bc", "omg", "literally", "trash", "thx"]
        slang_hits = sum(1 for t in all_texts if any(m in t.lower().split() for m in slang_markers))
        log.write(f"  Slang Density: {slang_hits/len(all_texts)*100:.1f}%\n")

    print("Full audit complete. Output written to full_dataset_audit_log.txt")

if __name__ == "__main__":
    analyze_full_dataset()
