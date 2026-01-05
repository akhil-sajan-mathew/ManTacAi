import json
import random

# --- MUTATION ENGINE ---
# This engine injects chaos to prevent "Synthetic Slop"
def mutate_text(text):
    # 1. Slang Injection
    replacements = {
        "you": ["u", "u", "you"], # 66% chance of slang
        "your": ["ur", "ur", "your"],
        "are": ["r", "r", "are"],
        "because": ["cuz", "coz", "bc", "cause", "because"],
        "literally": ["lit", "literally"],
        "love": ["luv", "love"],
        "hate": ["h8", "hate"],
        "please": ["plz", "pls", "please"]
    }
    
    words = text.split()
    new_words = []
    for w in words:
        clean_w = w.strip(".,!?")
        if clean_w.lower() in replacements and random.random() > 0.3:
            # Inject slang
            new_w = random.choice(replacements[clean_w.lower()])
            # Restore punctuation roughly
            if w.endswith("?"): new_w += "?"
            if w.endswith("!"): new_w += "!"
            new_words.append(new_w)
        else:
            new_words.append(w)
            
    text = " ".join(new_words)

    # 2. Case Chaos
    rand_case = random.random()
    if rand_case < 0.3:
        text = text.lower() # all lowercase (lazy typing)
    elif rand_case < 0.1:
        text = text.upper() # ALL CAPS RAGE
        
    # 3. Punctuation Noise
    if random.random() < 0.4:
        # Remove punctuation
        text = text.replace(".", "").replace(",", "").replace("'", "")
    
    # 4. Emotional Noise Injection (Prefix/Suffix)
    prefixes = ["omg", "ugh", "dude", "bro", "seriously", "wtf", "literally", "man", "yo"]
    suffixes = ["ffs", "smh", "stg", "fr", "tho", "rn", "lol", "u know", "ig"]
    
    if random.random() < 0.25:
        text = f"{random.choice(prefixes)} {text}"
    if random.random() < 0.25:
        text = f"{text} {random.choice(suffixes)}"

    return text.strip()

def generate_contrastive_dataset():
    data = []
    
    # --- BATCH 1: THE BLAME GAME (200 Items) ---
    # Safe Target vs Danger Target
    
    contexts = [
        "this game", "my job", "the wifi", "this server", "lag", "my boss", "traffic", 
        "this raid", "my team", "riot games", "apex", "valorant", "coding", "this project",
        "the internet", "my phone", "the app", "windows update"
    ]
    safe_targets = [
        "lag", "rng", "bad luck", "incompetence", "glitches", "bugs", "trash servers", 
        "stupid devs", "my controller", "packet loss", "fps drops"
    ]
    
    connectors = ["cuz of", "because of", "thanks to", "due to"]
    
    # Generate 100 SAFE (Benign Venting)
    for _ in range(100):
        ctx = random.choice(contexts)
        t = random.choice(safe_targets)
        c = random.choice(connectors)
        
        base = f"i hate {ctx} {c} {t}"
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "benign_venting", "category": "v7_blame_safe"})

    # Generate 100 DANGER (Ridicule/Coercion)
    # Using specific labels to boost under-represented classes
    for _ in range(100):
        ctx = random.choice(contexts)
        c = random.choice(connectors)
        
        # Danger Variants
        bad_base = random.choice([
            f"i hate {ctx} {c} you",
            f"i lost {ctx} {c} you",
            f"{ctx} is trash {c} u",
            f"im losing {ctx} {c} ur fault"
        ])
        
        final = mutate_text(bad_base)
        # We classify this as 'belittling_ridicule' or 'coercive_control' depending on intensity, 
        # but here we stick to 'belittling_ridicule' as "You are the problem" is belittling.
        data.append({"text": final, "manipulation_tactic": "belittling_ridicule", "category": "v7_blame_danger"})


    # --- BATCH 2: THERAPY SPEAK (200 Items) ---
    # Healthy Feeling vs Weaponized Feeling
    
    emotions = ["hurt", "sad", "unheard", "lonely", "overwhelmed", "anxious", "scared", "frustrated", "tired"]
    insults = ["crazy", "psycho", "paranoid", "too sensitive", "imagining things", "overreacting", "abusive", "controlling"]
    
    # Generate 100 SAFE (Healthy Conflict)
    for _ in range(100):
        emo = random.choice(emotions)
        base = random.choice([
            f"i feel {emo}",
            f"i feel {emo} right now",
            f"i am feeling {emo} about this",
            f"honestly i feel {emo}"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "healthy_conflict", "category": "v7_therapy_safe"})
        
    # Generate 100 DANGER (Gaslighting/Darvo)
    for _ in range(100):
        ins = random.choice(insults)
        base = random.choice([
            f"i feel like you are {ins}",
            f"i feel that u are being {ins}",
            f"i feel like ur acting {ins}",
            f"honestly i feel u are {ins}"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "gaslighting", "category": "v7_therapy_danger"})

    # --- SAVE ---
    output_path = "dataset_augmented/v7_nuance_400.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Generated {len(data)} items (Surgical V7) to {output_path}")
    print("Sample:", data[0]['text'])

if __name__ == "__main__":
    generate_contrastive_dataset()
