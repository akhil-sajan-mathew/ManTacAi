import json
import random
from datetime import datetime

def generate_gaslighting_examples():
    """Generate synthetic gaslighting examples"""
    templates = [
        "That never happened, you're imagining things.",
        "I never said that. You must be confused.",
        "You're being too sensitive, it wasn't that bad.",
        "Are you sure that's what happened? Your memory isn't very reliable.",
        "You're overreacting. That's not what I meant at all.",
        "I think you're misremembering the situation.",
        "You always twist my words. I never said anything like that.",
        "You're being paranoid. Nobody else sees it that way.",
        "That's not how it happened at all. You need to get your facts straight.",
        "You're making this up in your head. I would never do that."
    ]
    
    examples = []
    for i, template in enumerate(templates):
        examples.append({
            'id': f"gaslighting_{i}",
            'text': template,
            'manipulation_tactic': 'gaslighting',
            'severity_score': 75,
            'confidence': 0.95,
            'source': 'synthetic',
            'length': len(template),
            'created_at': datetime.now().isoformat()
        })
    
    return examples

def generate_guilt_tripping_examples():
    """Generate synthetic guilt-tripping examples"""
    templates = [
        "After everything I've done for you, this is how you repay me?",
        "I guess my feelings don't matter to you.",
        "Fine, don't worry about me. I'll just suffer in silence.",
        "I sacrificed so much for you, and you can't even do this one thing.",
        "You're being selfish. Think about how this affects me.",
        "I'm so disappointed in you. I expected better.",
        "If you really cared about me, you would do this.",
        "I can't believe you're abandoning me when I need you most.",
        "You're hurting me by not understanding my position.",
        "I thought you were different, but you're just like everyone else."
    ]
    
    examples = []
    for i, template in enumerate(templates):
        examples.append({
            'id': f"guilt_trip_{i}",
            'text': template,
            'manipulation_tactic': 'guilt_tripping',
            'severity_score': 65,
            'confidence': 0.9,
            'source': 'synthetic',
            'length': len(template),
            'created_at': datetime.now().isoformat()
        })
    
    return examples

def generate_deflection_examples():
    """Generate synthetic deflection examples"""
    templates = [
        "What about when you did the same thing last week?",
        "You're one to talk. Remember what you did?",
        "This isn't about me, it's about your behavior.",
        "Why are you bringing this up now? What's your real agenda?",
        "You're just trying to distract from your own mistakes.",
        "That's completely different and you know it.",
        "You're changing the subject because you know I'm right.",
        "This is typical of you to blame others.",
        "You always do this when you're caught.",
        "Nice try, but we're not talking about me here."
    ]
    
    examples = []
    for i, template in enumerate(templates):
        examples.append({
            'id': f"deflection_{i}",
            'text': template,
            'manipulation_tactic': 'deflection',
            'severity_score': 55,
            'confidence': 0.85,
            'source': 'synthetic',
            'length': len(template),
            'created_at': datetime.now().isoformat()
        })
    
    return examples

def generate_stonewalling_examples():
    """Generate synthetic stonewalling examples"""
    templates = [
        "I'm not discussing this anymore.",
        "Whatever. I'm done talking.",
        "Fine. Have it your way.",
        "I don't want to hear it.",
        "This conversation is over.",
        "I'm not listening to this.",
        "Talk to the wall.",
        "I have nothing more to say.",
        "You can keep talking, but I'm not responding.",
        "I'm shutting down this discussion."
    ]
    
    examples = []
    for i, template in enumerate(templates):
        examples.append({
            'id': f"stonewalling_{i}",
            'text': template,
            'manipulation_tactic': 'stonewalling',
            'severity_score': 60,
            'confidence': 0.8,
            'source': 'synthetic',
            'length': len(template),
            'created_at': datetime.now().isoformat()
        })
    
    return examples

def generate_love_bombing_examples():
    """Generate synthetic love-bombing examples"""
    templates = [
        "You're absolutely perfect in every way. I've never met anyone like you.",
        "I can't live without you. You mean everything to me.",
        "You're the most amazing person I've ever known. I worship you.",
        "I love you more than life itself. You're my everything.",
        "You're so special and unique. Nobody understands you like I do.",
        "I would do anything for you. You're my whole world.",
        "You're incredibly talented and beautiful. I'm so lucky to have you.",
        "I've never felt this way about anyone. You're my soulmate.",
        "You're absolutely brilliant. I admire everything about you.",
        "You're the best thing that ever happened to me. I adore you completely."
    ]
    
    examples = []
    for i, template in enumerate(templates):
        examples.append({
            'id': f"love_bombing_{i}",
            'text': template,
            'manipulation_tactic': 'love_bombing',
            'severity_score': 70,
            'confidence': 0.85,
            'source': 'synthetic',
            'length': len(template),
            'created_at': datetime.now().isoformat()
        })
    
    return examples

def generate_appeal_to_emotion_examples():
    """Generate synthetic appeal to emotion examples"""
    templates = [
        "Think of the children! How can you be so heartless?",
        "This is a matter of life and death! You have to help!",
        "I'm devastated and heartbroken. Please reconsider.",
        "You're destroying everything I've worked for!",
        "This is the most important thing in my life!",
        "I'm begging you with tears in my eyes.",
        "My heart is breaking just thinking about this.",
        "This is an emergency! You must act now!",
        "I'm suffering so much because of this decision.",
        "Please, for the love of everything sacred!"
    ]
    
    examples = []
    for i, template in enumerate(templates):
        examples.append({
            'id': f"appeal_emotion_{i}",
            'text': template,
            'manipulation_tactic': 'appeal_to_emotion',
            'severity_score': 50,
            'confidence': 0.8,
            'source': 'synthetic',
            'length': len(template),
            'created_at': datetime.now().isoformat()
        })
    
    return examples

def generate_whataboutism_examples():
    """Generate synthetic whataboutism examples"""
    templates = [
        "What about when you did the exact same thing?",
        "But what about your behavior last month?",
        "You're criticizing me, but what about your mistakes?",
        "That's rich coming from someone who...",
        "Before you judge me, what about your own actions?",
        "You have no right to complain when you...",
        "What about all the times you've done worse?",
        "You're being hypocritical. What about when you...",
        "I may have done that, but what about you?",
        "You can't talk when you've done the same thing."
    ]
    
    examples = []
    for i, template in enumerate(templates):
        examples.append({
            'id': f"whataboutism_{i}",
            'text': template,
            'manipulation_tactic': 'whataboutism',
            'severity_score': 45,
            'confidence': 0.85,
            'source': 'synthetic',
            'length': len(template),
            'created_at': datetime.now().isoformat()
        })
    
    return examples

def create_complete_dataset():
    """Combine real and synthetic examples into complete dataset"""
    
    # Load existing clean dataset
    with open('clean_manipulation_dataset.json', 'r', encoding='utf-8') as f:
        real_examples = json.load(f)
    
    print(f"Loaded {len(real_examples)} real examples")
    
    # Generate synthetic examples
    synthetic_examples = []
    synthetic_examples.extend(generate_gaslighting_examples())
    synthetic_examples.extend(generate_guilt_tripping_examples())
    synthetic_examples.extend(generate_deflection_examples())
    synthetic_examples.extend(generate_stonewalling_examples())
    synthetic_examples.extend(generate_love_bombing_examples())
    synthetic_examples.extend(generate_appeal_to_emotion_examples())
    synthetic_examples.extend(generate_whataboutism_examples())
    
    print(f"Generated {len(synthetic_examples)} synthetic examples")
    
    # Combine datasets
    complete_dataset = real_examples + synthetic_examples
    
    # Create final summary
    tactics_count = {}
    for example in complete_dataset:
        tactic = example['manipulation_tactic']
        tactics_count[tactic] = tactics_count.get(tactic, 0) + 1
    
    summary = {
        'total_examples': len(complete_dataset),
        'real_examples': len(real_examples),
        'synthetic_examples': len(synthetic_examples),
        'tactics_distribution': tactics_count,
        'coverage': {
            'all_10_tactics_covered': len(tactics_count) == 10,
            'tactics_covered': list(tactics_count.keys())
        }
    }
    
    return complete_dataset, summary

if __name__ == "__main__":
    complete_dataset, summary = create_complete_dataset()
    
    # Save complete dataset
    with open('complete_manipulation_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(complete_dataset, f, indent=2, ensure_ascii=False)
    
    # Save summary
    with open('complete_dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== COMPLETE DATASET CREATED ===")
    print(f"Total examples: {summary['total_examples']}")
    print(f"Real examples: {summary['real_examples']}")
    print(f"Synthetic examples: {summary['synthetic_examples']}")
    
    print(f"\nAll 10 tactics covered: {summary['coverage']['all_10_tactics_covered']}")
    print(f"\nTactics distribution:")
    for tactic, count in summary['tactics_distribution'].items():
        print(f"  {tactic}: {count}")
    
    print(f"\nFiles created:")
    print(f"  - complete_manipulation_dataset.json ({len(complete_dataset)} examples)")
    print(f"  - complete_dataset_summary.json")
    
    print(f"\n=== READY FOR MODEL TRAINING ===")
    print("Next steps:")
    print("1. Review and validate synthetic examples")
    print("2. Create train/validation/test splits")
    print("3. Begin model training with BERT/RoBERTa")
    print("4. Implement multi-label classification pipeline")