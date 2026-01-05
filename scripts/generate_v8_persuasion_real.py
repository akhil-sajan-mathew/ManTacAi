import json
import random

# --- MUTATION ENGINE ---
# Vital for removing the "AI Slop" feel.
def mutate_text(text):
    replacements = {
        "you": ["u", "u", "you"], 
        "are": ["r", "r", "are"],
        "because": ["cuz", "coz", "bc"],
        "please": ["plz", "pls"],
        "really": ["rly", "really"],
        "think": ["thnk"],
        "would": ["wd"],
        "could": ["cd"]
    }
    
    words = text.split()
    new_words = []
    for w in words:
        clean_w = w.strip(".,!?")
        if clean_w.lower() in replacements and random.random() > 0.35:
            new_w = random.choice(replacements[clean_w.lower()])
            if w.endswith("?"): new_w += "?"
            new_words.append(new_w)
        else:
            new_words.append(w)
            
    text = " ".join(new_words)

    # 40% Chance: Lowercase everything (Lazy Typer)
    if random.random() < 0.4:
        text = text.lower()
        
    # 20% Chance: Remove final punctuation
    if random.random() < 0.2 and text.endswith("."):
        text = text[:-1]

    return text.strip()

def generate_persuasion_real():
    data = []
    
    # =========================================================================
    # THE "ETHICAL PERSUASION" ENGINE (800 Unique Items)
    # Goal: Dialogue that tries to convince WITHOUT manipulation/threats.
    # Key Markers: "I think", "Maybe", "What if", "It would help".
    # =========================================================================
    
    # 1. RELATIONSHIP GROWTH (200)
    # Trying to convince partner to do something healthy (Therapy, Date Night, Chores)
    starts = ["i really think", "hey can we", "it would mean a lot if", "honestly i feel like"]
    topics = ["tried therapy", "went on a date", "saved money", "talked more", "visited my parents"]
    reasons = ["it would help us", "we need a break", "it brings us closer", "it solves the stress"]
    
    for _ in range(200):
        s = random.choice(starts)
        t = random.choice(topics)
        r = random.choice(reasons)
        
        # Structure: Start + Topic + Reason (Softened)
        base = f"{s} we should {t} becase {r}"
        
        # Mutation: High variance
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "ethical_persuasion", "category": "v8_persuasion_relationship"})

    # 2. SOCIAL & PLANS (200)
    # Convincing friends/partner to go out or stay in.
    events = ["party", "movie", "dinner", "trip", "concert", "hike"]
    benefits = ["it will be fun", "everyone is going", "you need a break", "the food is good", "ticks are cheap"]
    appeals = ["come on", "please just consider it", "give it a shot", "trust me"]
    
    for _ in range(200):
        e = random.choice(events)
        b = random.choice(benefits)
        a = random.choice(appeals)
        
        base = random.choice([
            f"{a} let's go to the {e} {b}",
            f"i know you are tired but the {e} {b}",
            f"what if we just go to the {e} for an hour? {b}",
            f"{b} so {a} let's do the {e}"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "ethical_persuasion", "category": "v8_persuasion_social"})

    # 3. WORK & CAREER (200)
    # Professional negotiation (Asking for raise, deadline extension, new tool)
    requests = ["a raise", "more time", "help with this", "a new laptop", "to lead this project"]
    biz_reasons = ["i worked hard", "the quality will be better", "it is efficient", "i have the experience"]
    polite_openers = ["could we discuss", "i was wondering if", "i believe i deserve", "my proposal is"]
    
    for _ in range(200):
        req = random.choice(requests)
        br = random.choice(biz_reasons)
        po = random.choice(polite_openers)
        
        base = f"{po} getting {req} because {br}"
        # Less slang for work, but strict lowercase is common in slack
        final = mutate_text(base) 
        data.append({"text": final, "manipulation_tactic": "ethical_persuasion", "category": "v8_persuasion_work"})

    # 4. HOUSEHOLD NEGOTIATION (200)
    # Boring but real conflicts (Budget, Paint color, Dinner choice)
    issues = ["budget", "paint color", "dinner", "vacation spot", "internet plan"]
    solutions = ["compromise", "split the difference", "try my way", "look at the data"]
    softeners = ["i understand but", "hear me out", "i get your point but", "let's look at facts"]
    
    for _ in range(200):
        i = random.choice(issues)
        sol = random.choice(solutions)
        soft = random.choice(softeners)
        
        base = f"{soft} about the {i} maybe we can {sol}"
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "ethical_persuasion", "category": "v8_persuasion_household"})

    # Save
    output_path = "dataset_augmented/v8_persuasion_real_800.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Generated {len(data)} REAL Persuasion items to {output_path}")
    print("Sample:", data[0]['text'])

if __name__ == "__main__":
    generate_persuasion_real()
