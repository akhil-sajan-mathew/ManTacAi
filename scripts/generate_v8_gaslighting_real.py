import json
import random

# --- MUTATION ENGINE (The Anti-Slop Filter) ---
def mutate_text(text):
    replacements = {
        "you": ["u", "u", "you"], 
        "are": ["r", "r", "are"],
        "your": ["ur", "ur"],
        "because": ["cuz"],
        "please": ["plz"],
        "crazy": ["cray", "insane", "nuts", "psycho"],
        "seriously": ["srsly"],
        "joking": ["jk", "kidding"],
        "what": ["wat", "wht"]
    }
    
    words = text.split()
    new_words = []
    for w in words:
        clean_w = w.strip(".,!?")
        if clean_w.lower() in replacements and random.random() > 0.4:
            new_w = random.choice(replacements[clean_w.lower()])
            if w.endswith("?"): new_w += "?"
            new_words.append(new_w)
        else:
            new_words.append(w)
            
    text = " ".join(new_words)

    # 40% chance: Lowercase (Casual dismissiveness)
    if random.random() < 0.4:
        text = text.lower()
        
    return text.strip()

def generate_gaslighting_real():
    data = []
    
    # =========================================================================
    # THE "REAL" GASLIGHTING ENGINE (600 Items)
    # Goal: Dialogue sent BY the abuser to the victim.
    # No internal monologues. No "villain speeches". Just toxic texting.
    # =========================================================================
    
    # 1. REALITY DENIAL (150)
    # "I never said that."
    phrases = ["i never said that", "that literally never happened", "you are imagining things", "stop making stuff up"]
    closers = ["check your memory", "you always do this", "you are confused", "stop lying"]
    
    for _ in range(150):
        p = random.choice(phrases)
        c = random.choice(closers)
        base = f"{p} {c}"
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "gaslighting", "category": "v8_gaslighting_denial"})

    # 2. PATHOLOGIZING / INSULTS (150)
    # "You are crazy/sensitive."
    labels = ["crazy", "bipolar", "psycho", "paranoid", "too sensitive", "hysterical", "delusional"]
    openers = ["omg you are so", "why are you acting", "you sound", "stop being so"]
    
    for _ in range(150):
        l = random.choice(labels)
        o = random.choice(openers)
        base = random.choice([
            f"{o} {l} right now",
            f"everyone knows you are {l}",
            f"you need help being this {l}",
            f"calm down you are acting {l}"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "gaslighting", "category": "v8_gaslighting_insult"})

    # 3. TRIVIALIZING (150)
    # "It was just a joke."
    excuses = ["just a joke", "only kidding", "being sarcastic", "messing with you"]
    blame = ["take a joke", "get a sense of humor", "lighten up", "stop being dramatic"]
    
    for _ in range(150):
        e = random.choice(excuses)
        b = random.choice(blame)
        base = f"it was {e} learn to {b}"
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "gaslighting", "category": "v8_gaslighting_trivial"})

    # 4. HISTORY REWRITING (150)
    # "You agreed to this."
    claims = ["you agreed to this", "we talked about this", "you said it was fine", "you literally purposed this"]
    memory_shame = ["did you forget?", "is your memory that bad?", "why do you forget everything?", "i remember distinctively"]
    
    for _ in range(150):
        c = random.choice(claims)
        m = random.choice(memory_shame)
        base = f"{c} {m}"
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "gaslighting", "category": "v8_gaslighting_history"})

    # Save
    output_path = "dataset_augmented/v8_gaslighting_real_600.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Generated {len(data)} REAL Gaslighting items to {output_path}")
    print("Sample:", data[0]['text'])

if __name__ == "__main__":
    generate_gaslighting_real()
