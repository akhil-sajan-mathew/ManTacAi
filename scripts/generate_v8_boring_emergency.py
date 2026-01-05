import json
import random

# --- MUTATION ENGINE (Consistency is Key) ---
def mutate_text(text):
    replacements = {
        "you": ["u", "u", "you"], 
        "are": ["r", "r", "are"],
        "because": ["cuz", "coz", "bc"],
        "please": ["plz", "pls"],
        "okay": ["k", "ok", "kk"],
        "right now": ["rn"],
        "tomorrow": ["tmrw"],
        "tonight": ["tn", "2nite"],
        "meeting": ["mtg"],
        "email": ["mail"],
        "battery": ["batt"],
        "phone": ["cell"]
    }
    
    words = text.split()
    new_words = []
    for w in words:
        clean_w = w.strip(".,!?")
        if clean_w.lower() in replacements and random.random() > 0.4:
            new_w = random.choice(replacements[clean_w.lower()])
            if w.endswith("?"): new_w += "?"
            if w.endswith("!"): new_w += "!"
            new_words.append(new_w)
        else:
            new_words.append(w)
            
    text = " ".join(new_words)

    # 50% chance to remove punctuation for "lazy texting"
    if random.random() < 0.5:
        text = text.replace(".", "").replace(",", "").replace("'", "")
        
    # 60% chance to lowercase everything
    if random.random() < 0.6:
        text = text.lower()
        
    return text.strip()

def generate_v8_dataset():
    data = []
    
    # =========================================================================
    # PART 1: THE "BORING" SURGE (1,000 Items)
    # Goal: Substantive Neutrality (>3 words)
    # =========================================================================
    
    # Scenario A: Household Chores (300)
    chores = ["laundry", "dishes", "trash", "recycling", "groceries", "cleaning", "vacuuming", "mowing"]
    statuses = ["done", "finished", "started", "doing it now", "forgot to do", "need to do"]
    
    for _ in range(300):
        c = random.choice(chores)
        s = random.choice(statuses)
        base = random.choice([
            f"i just {s} the {c}",
            f"did you get the {c} done?",
            f"please remember the {c}",
            f"i am taking out the {c} now",
            f"can you help with {c} later?"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "neutral_logistics", "category": "v8_boring_chores"})

    # Scenario B: Work Fatigue (350)
    work_tasks = ["email", "report", "spreadsheet", "meeting", "call", "zoom", "slack", "presentation"]
    work_times = ["until 5", "all day", "in 10 mins", "at 2pm", "later"]
    
    for _ in range(350):
        w = random.choice(work_tasks)
        t = random.choice(work_times)
        base = random.choice([
            f"stuck in a {w} {t}",
            f"just sent that {w}",
            f"can i call you after this {w}?",
            f"my {w} is running late",
            f"busy with {w} right now",
            f"finish your {w} yet?"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "neutral_logistics", "category": "v8_boring_work"})

    # Scenario C: Digital/Life Logistics (350)
    tech_issues = ["battery", "charger", "wifi", "signal", "laptop", "car", "keys", "wallet"]
    actions = ["dying", "dead", "lost", "forgot", "charging", "looking for"]
    
    for _ in range(350):
        t = random.choice(tech_issues)
        a = random.choice(actions)
        base = random.choice([
            f"my {t} is {a} right now",
            f"have you seen my {t}?",
            f"waiting for my {t} to charge",
            f"i think i left my {t} there",
            f"fixing the {t} real quick",
            f"brb my {t} died"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "neutral_logistics", "category": "v8_boring_logistics"})


    # =========================================================================
    # PART 2: THE "EMERGENCY" BOOST (300 Items)
    # Goal: High Urgency, High Specificity
    # =========================================================================
    
    crises = ["hospital", "ER", "emergency room", "accident", "crash", "fire", "police", "ambulance", "doctor", "school nurse"]
    urgency = ["pick up the phone", "call me now", "answer me", "hurry up", "come home now", "need help"]
    reasons = ["bleeding", "fell", "hit head", "broke leg", "scared", "shaking", "smoke", "flooded", "leaking"]
    
    for _ in range(300):
        c = random.choice(crises)
        u = random.choice(urgency)
        r = random.choice(reasons)
        
        # High randomness for panic simulation
        base = random.choice([
            f"{u} i am at the {c}",
            f"please {u} it is an emergency",
            f"mom is at the {c}",
            f"i think i {r} please help",
            f"{c} just called {u}",
            f"there is a {c} {u}",
            f"seriously {u} i am {r}"
        ])
        
        # Less mutation for emergencies (people type fast but auto-correct helps, or they type chaotic)
        # We will use the standard mutation but maybe add CAPS sometimes
        final = mutate_text(base)
        if random.random() < 0.3:
            final = final.upper() # CAPS LOCK PANIC
            
        data.append({"text": final, "manipulation_tactic": "urgent_emergency", "category": "v8_emergency"})

    # Save
    output_path = "dataset_augmented/v8_boring_emergency_1300.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Generated {len(data)} V8 Items to {output_path}")
    print("Sample Boring:", data[0]['text'])
    print("Sample Emergency:", data[-1]['text'])

if __name__ == "__main__":
    generate_v8_dataset()
