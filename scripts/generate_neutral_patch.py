import json
import random

# --- MUTATION ENGINE (Reused for Consistency) ---
def mutate_text(text):
    # slangs
    replacements = {
        "you": ["u", "u", "you"], 
        "are": ["r", "r", "are"],
        "because": ["cuz", "coz", "bc"],
        "please": ["plz", "pls"],
        "okay": ["k", "ok", "kk"],
        "thanks": ["thx", "ty"],
        "tonight": ["tn", "2nite"],
        "tomorrow": ["tmrw", "tom"],
        "right now": ["rn"],
        "what": ["wat", "wht"],
        "have": ["hv"]
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

    # punc removal
    if random.random() < 0.5:
        text = text.replace(".", "").replace(",", "").replace("'", "")
        
    # lowercase
    if random.random() < 0.6:
        text = text.lower()
        
    return text.strip()

def generate_neutral_patch():
    data = []
    
    # --- SCENARIO 1: LOGISTICS (Adulting) ---
    actions = ["paying", "paid", "checking", "sent"]
    bills = ["gas bill", "electric", "rent", "internet", "netflix", "water bill"]
    
    for _ in range(125):
        act = random.choice(actions)
        bill = random.choice(bills)
        base = random.choice([
            f"did you pay the {bill}?",
            f"i just paid the {bill}",
            f"can you check the {bill}?",
            f"reminder to pay {bill}",
            f"sent the money for {bill}"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "neutral_conversation", "category": "v7_patch_logistics"})

    # --- SCENARIO 2: COORDINATION (ETA) ---
    times = ["5 mins", "10 mins", "soon", "later", "at 6", "tmrw"]
    places = ["home", "work", "gym", "store", "school"]
    
    for _ in range(125):
        t = random.choice(times)
        p = random.choice(places)
        base = random.choice([
            f"i will be {p} in {t}",
            f"are you at {p} yet?",
            f"leaving {p} now",
            f"stuck in traffic be there {t}",
            f"can you pick me up from {p}?"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "neutral_conversation", "category": "v7_patch_eta"})

    # --- SCENARIO 3: FOOD (Hunger) ---
    foods = ["pizza", "tacos", "sushi", "burgers", "leftovers", "eggs", "milk", "bread"]
    
    for _ in range(125):
        f = random.choice(foods)
        base = random.choice([
            f"what is for dinner?",
            f"do we have {f}?",
            f"i am ordering {f}",
            f"can you buy {f}?",
            f"i am hungry for {f}",
            f"we need more {f}"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "neutral_conversation", "category": "v7_patch_food"})

    # --- SCENARIO 4: MEDIA/OBSERVATION ---
    things = ["movie", "show", "game", "weather", "song", "video"]
    adjs = ["good", "bad", "boring", "crazy", "funny", "weird"]
    
    for _ in range(125):
        th = random.choice(things)
        adj = random.choice(adjs)
        base = random.choice([
            f"did you see that {th}?",
            f"this {th} is {adj}",
            f"watching a {th} rn",
            f"look at this {th}",
            f"that {th} was {adj}"
        ])
        final = mutate_text(base)
        data.append({"text": final, "manipulation_tactic": "neutral_conversation", "category": "v7_patch_media"})

    # Save
    output_path = "dataset_augmented/v7_neutral_patch_500.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Generated {len(data)} Neutral Items to {output_path}")
    print("Sample:", data[0]['text'])

if __name__ == "__main__":
    generate_neutral_patch()
