import json
import pandas as pd
from collections import Counter

def load_and_analyze_dataset():
    """Load and provide detailed analysis of the complete dataset"""
    
    with open('complete_manipulation_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    df = pd.DataFrame(dataset)
    
    print("=== MANIPULATION DETECTION DATASET ANALYSIS ===\n")
    
    # Basic statistics
    print(f"📊 DATASET OVERVIEW")
    print(f"Total examples: {len(dataset)}")
    print(f"Real examples: {len([d for d in dataset if d['source'] != 'synthetic'])}")
    print(f"Synthetic examples: {len([d for d in dataset if d['source'] == 'synthetic'])}")
    
    # Tactics distribution
    print(f"\n🎯 MANIPULATION TACTICS COVERAGE")
    tactics_count = Counter([d['manipulation_tactic'] for d in dataset])
    for tactic, count in sorted(tactics_count.items()):
        percentage = (count / len(dataset)) * 100
        print(f"  {tactic.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Quality metrics
    print(f"\n📈 QUALITY METRICS")
    confidences = [d['confidence'] for d in dataset]
    severities = [d['severity_score'] for d in dataset]
    lengths = [d['length'] for d in dataset]
    
    print(f"  Average confidence: {sum(confidences)/len(confidences):.2f}")
    print(f"  Average severity: {sum(severities)/len(severities):.1f}")
    print(f"  Average text length: {sum(lengths)/len(lengths):.1f} characters")
    print(f"  Length range: {min(lengths)} - {max(lengths)} characters")
    
    # Examples needing review
    needs_review = [d for d in dataset if d.get('needs_manual_review', False)]
    print(f"  Examples needing manual review: {len(needs_review)}")
    
    return dataset, df

def show_examples_by_tactic(dataset, tactic, num_examples=3):
    """Show sample examples for a specific tactic"""
    
    tactic_examples = [d for d in dataset if d['manipulation_tactic'] == tactic]
    
    if not tactic_examples:
        print(f"No examples found for {tactic}")
        return
    
    print(f"\n🔍 SAMPLE EXAMPLES: {tactic.replace('_', ' ').upper()}")
    print(f"Total examples: {len(tactic_examples)}")
    
    # Show mix of real and synthetic
    real_examples = [d for d in tactic_examples if d['source'] != 'synthetic']
    synthetic_examples = [d for d in tactic_examples if d['source'] == 'synthetic']
    
    if real_examples:
        print(f"\n  Real Examples:")
        for i, example in enumerate(real_examples[:num_examples]):
            print(f"    {i+1}. \"{example['text'][:100]}{'...' if len(example['text']) > 100 else ''}\"")
            print(f"       (Confidence: {example['confidence']}, Severity: {example['severity_score']})")
    
    if synthetic_examples:
        print(f"\n  Synthetic Examples:")
        for i, example in enumerate(synthetic_examples[:num_examples]):
            print(f"    {i+1}. \"{example['text'][:100]}{'...' if len(example['text']) > 100 else ''}\"")
            print(f"       (Confidence: {example['confidence']}, Severity: {example['severity_score']})")

def identify_data_gaps(dataset):
    """Identify potential gaps in the dataset"""
    
    print(f"\n⚠️  DATA QUALITY ASSESSMENT")
    
    tactics_count = Counter([d['manipulation_tactic'] for d in dataset])
    
    # Identify underrepresented tactics
    min_examples = 50  # Minimum recommended examples per tactic
    underrepresented = []
    
    for tactic, count in tactics_count.items():
        if count < min_examples:
            underrepresented.append((tactic, count))
    
    if underrepresented:
        print(f"\n  🔴 Underrepresented tactics (< {min_examples} examples):")
        for tactic, count in underrepresented:
            print(f"    {tactic.replace('_', ' ').title()}: {count} examples")
        print(f"    Recommendation: Generate more synthetic examples or find real examples")
    
    # Check for imbalanced dataset
    max_count = max(tactics_count.values())
    min_count = min(tactics_count.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n  📊 Dataset Balance:")
    print(f"    Imbalance ratio: {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 10:
        print(f"    🔴 High imbalance detected - consider balancing techniques")
    elif imbalance_ratio > 5:
        print(f"    🟡 Moderate imbalance - monitor model performance")
    else:
        print(f"    🟢 Reasonable balance")
    
    # Check confidence distribution
    low_confidence = [d for d in dataset if d['confidence'] < 0.7]
    print(f"\n  🎯 Confidence Distribution:")
    print(f"    Low confidence examples (< 0.7): {len(low_confidence)}")
    if len(low_confidence) > len(dataset) * 0.2:
        print(f"    🔴 High number of low-confidence examples - review quality")
    else:
        print(f"    🟢 Confidence levels look good")

def create_training_splits(dataset):
    """Create train/validation/test splits"""
    
    import random
    random.seed(42)
    
    # Shuffle dataset
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    
    # Create splits (70% train, 15% val, 15% test)
    total = len(shuffled)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    
    train_data = shuffled[:train_size]
    val_data = shuffled[train_size:train_size + val_size]
    test_data = shuffled[train_size + val_size:]
    
    # Save splits
    splits = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    with open('dataset_splits.json', 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    
    print(f"\n📂 DATASET SPLITS CREATED")
    print(f"  Training: {len(train_data)} examples ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation: {len(val_data)} examples ({len(val_data)/total*100:.1f}%)")
    print(f"  Test: {len(test_data)} examples ({len(test_data)/total*100:.1f}%)")
    print(f"  Saved to: dataset_splits.json")
    
    return splits

if __name__ == "__main__":
    # Load and analyze dataset
    dataset, df = load_and_analyze_dataset()
    
    # Show examples for each tactic
    tactics = [
        'gaslighting', 'guilt_tripping', 'deflection', 'stonewalling',
        'belittling_ridicule', 'love_bombing', 'threatening_intimidation',
        'passive_aggression', 'appeal_to_emotion', 'whataboutism'
    ]
    
    for tactic in tactics:
        show_examples_by_tactic(dataset, tactic, 2)
    
    # Identify data gaps
    identify_data_gaps(dataset)
    
    # Create training splits
    splits = create_training_splits(dataset)
    
    print(f"\n✅ DATASET READY FOR MODEL TRAINING")
    print(f"Next steps:")
    print(f"1. Review low-confidence examples manually")
    print(f"2. Generate more examples for underrepresented tactics")
    print(f"3. Implement BERT/RoBERTa multi-label classifier")
    print(f"4. Train model using dataset_splits.json")