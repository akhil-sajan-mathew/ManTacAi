import json
import random
import os

# --- CRITICAL INSTRUCTIONS IMPLEMENTATION ---

def apply_realism(text):
    """
    Applies slang, typos, and format degradation to make text feel real.
    CRITICAL INSTRUCTION #1: Realism
    """
    # 1. Lowercase (50% chance)
    if random.random() < 0.5:
        text = text.lower()

    # 2. Slang Swaps (Partial)
    slang_map = {
        "you": "u",
        "are": "r",
        "to": "2",
        "for": "4",
        "because": "cuz",
        "don't": "dont",
        "know": "kno",
        "going to": "gonna",
        "want to": "wanna",
        "okay": "k",
        "thanks": "thx",
        "please": "plz",
        "people": "ppl",
        "really": "rly",
        "wait": "w8"
    }
    
    words = text.split()
    new_words = []
    for w in words:
        clean_w = w.strip(".,!?").lower()
        if clean_w in slang_map and random.random() < 0.4: # 40% chance to slangify
            new_words.append(slang_map[clean_w])
        else:
            new_words.append(w)
    
    text = " ".join(new_words)

    # 3. Punctuation Stripping (30% chance)
    if random.random() < 0.3:
        text = text.replace(".", "").replace(",", "").replace("!", "").replace("?", "")

    # 4. Typos (Rare - 5% per char logic is too heavy, let's just swap random chars occasionally)
    if random.random() < 0.2: # 20% of messages get a typo
        if len(text) > 5:
            idx = random.randint(0, len(text)-2)
            # Swap two chars
            text = text[:idx] + text[idx+1] + text[idx] + text[idx+2:]

    return text

def generate_batch(category_name, definitions, count=250):
    """
    Generates 'count' examples for a specific category obeying the length distribution.
    CRITICAL INSTRUCTION #2: Variety (20% Short, 40% Medium, 40% Long)
    """
    dataset = []
    
    short_count = int(count * 0.2)
    medium_count = int(count * 0.4)
    long_count = count - short_count - medium_count # Remainder
    
    counts = {"short": short_count, "medium": medium_count, "long": long_count}
    
    for length_type, n in counts.items():
        for _ in range(n):
            base_text = random.choice(definitions[length_type])
            final_text = apply_realism(base_text)
            
            entry = {
                "text": final_text,
                "manipulation_tactic": category_name,
                "severity_score": 0,
                "confidence": 1.0,
                "is_manipulation": False,
                "source": "v6_normalcy_1250"
            }
            dataset.append(entry)
            
    random.shuffle(dataset)
    return dataset

# --- TEMPLATE DEFINITIONS (SAFE & REALISTIC) ---

TEMPLATES = {
    # BATCH 1: BENIGN VENTING (Boss, Traffic, Games)
    "benign_venting": {
        "short": ["ugh", "so mad", "furious", "hate this", "why", "literally dying", "cant even", "omg", "stupid", "ffs"],
        "medium": [
            "My boss is such a jerk today.",
            "I literally hate this traffic so much.",
            "This game is so stupid, I keep losing.",
            "Why is the internet so slow right now?",
            "I am so done with this project.",
            "My team is absolutely useless.",
            "I swear I'm going to quit this job.",
            "The ref is completely blind.",
            "I can't believe they cancelled my show.",
            "My computer just crashed again."
        ],
        "long": [
            "My manager is making me redo the entire report for no reason. I am actually going to lose my mind.",
            "Dude this lag is unplayable. I swear the servers are run by hamsters. I hate this game.",
            "Stuck in traffic for 2 hours now. People clearly don't know how to drive. I'm gonna scream.",
            "I tried to fix the sink but now it's leaking worse. I am so frustrated right now I could punch a wall.",
            "Why does the printer always break when I need it? I swear technology hates me personally.",
            "Political twitter is melting down again. I hate reading this garbage but I can't look away.",
            "My favorite sports team just threw the game in the last minute. I am absolutely furious.",
            "The customer service line hung up on me after 40 minutes on hold. I want to throw my phone.",
            "My delivery is 3 hours late and cold. I am leaving the worst review ever.",
            "Coding this feature is a nightmare. Nothing works and I hate everything about this language."
        ]
    },

    # BATCH 2: HEALTHY CONFLICT (Boundaries, No)
    "healthy_conflict": {
        "short": ["no", "cant", "nope", "i disagree", "stop", "not today", "id rather not", "pass", "nah", "sorry no"],
        "medium": [
            "I can't come over tonight, I'm tired.",
            "I don't agree with that opinion.",
            "Please stop doing that, it bothers me.",
            "I need some space right now.",
            "I'm not comfortable with that decision.",
            "We should probably take a break.",
            "I'm going to stay home this weekend.",
            "That's not what I said.",
            "I don't think this is working out.",
            "You need to ask before taking my stuff."
        ],
        "long": [
            "I understand you're upset, but I'm not going to argue about this right now. We can talk later.",
            "I've told you I don't like horror movies. Please stop trying to make me watch them.",
            "I'm really busy with work this week so I can't handle the chores. You need to do them.",
            "I appreciate the invite but I really need a mental health day. I'm staying in.",
            "That joke wasn't funny, it was actually kind of rude. Please don't say that again.",
            "I disagree with your political take. Let's just agree to not talk about this topic.",
            "I'm not going to lend you money again until you pay me back for the last time.",
            "I need you to respect my privacy and not read my phone notifications.",
            "I'm feeling overwhelmed so I'm going for a walk alone. Do not follow me.",
            "This isn't a debate. I said no and I mean it. Please respect my answer."
        ]
    },
    
    # BATCH 3: BENIGN AFFECTION (Love, Thanks)
    "benign_affection": {
        "short": ["love u", "thx", "miss u", "so happy", "bestie", "great job", "ur hot", "yay", "<3", "cute"],
        "medium": [
            "Thanks for making dinner, it was great.",
            "You look really nice in that outfit.",
            "I missed you today.",
            "So excited to see you later!",
            "You are the best, seriously.",
            "Can't wait for our trip.",
            "I had a really good time tonight.",
            "Good luck with your presentation!",
            "I bought you that snack you like.",
            "Drive safe, text me when you get there."
        ],
        "long": [
            "Hey just wanted to say thank you for listening to me vent earlier. You're a great listener.",
            "I saw this meme and thought of you immediately. Miss your face!",
            "So proud of you for getting that promotion! You worked so hard for it.",
            "Happy anniversary babe. Actually can't believe it's been 2 years. Love you.",
            "Thanks for picking up the dry cleaning even though you were busy. Appreciate it.",
            "You are literally the funniest person I know. My cheeks hurt from laughing.",
            "Take a nap, you deserve it. I'll handle the dinner prep tonight.",
            "Just a reminder that you are awesome and you got this. Don't stress.",
            "Can we stay in and cuddle/watch movies tonight? I missed you this week.",
            "Your hair looks amazing today by the way. Did you do something different?"
        ]
    },

    # BATCH 4: NEUTRAL LOGISTICS (Chores, Time)
    "neutral_logistics": {
        "short": ["ok", "omw", "eta 5m", "got it", "where?", "whats the code", "ready?", "buying milk", "done", "feed cat"],
        "medium": [
            "Can you pick up milk on your way home?",
            "What time is our reservation?",
            "I paid the electric bill today.",
            "Don't forget to take the trash out.",
            "The plumber is coming at 2pm.",
            "I'll be there in 15 minutes.",
            "Did you feed the dog yet?",
            "We need to buy more laundry detergent.",
            "Send me the address please.",
            "Is the dishwasher clean or dirty?"
        ],
        "long": [
            "Hey, the car is making a weird noise again. We should probably take it to the shop.",
            "I'm going to be late because of traffic. Start eating without me.",
            "Please remember to sign that document and leave it on the counter.",
            "What should we get for your mom's birthday? I was thinking maybe flowers.",
            "I left the leftovers in the fridge for you. Heat them up for 2 minutes.",
            "Can you double check if I locked the front door? I'm paranoid.",
            "The internet bill is due on Friday. Can you transfer me your half?",
            "Grocery list: eggs, bread, spicy salsa, and those chips I like.",
            "I'm folding laundry right now. Can you come help me with the sheets?",
            "Schedule update: I have a meeting until 5, then gym. Home by 7."
        ]
    },

    # BATCH 5: URGENT EMERGENCY (Panic, Safety)
    "urgent_emergency": {
        "short": ["HELP", "CALL 911", "ACCIDENT", "FIRE", "HURT", "ANSWER ME", "PICK UP", "OMG", "BLOOD", "SCARED"],
        "medium": [
            "CALL ME RIGHT NOW IT'S AN EMERGENCY.",
            "I just got into a car accident.",
            "Mom is in the hospital.",
            "There's a fire in the kitchen!",
            "I think I broke my arm.",
            "Someone is trying to break in.",
            "The dog just ate chocolate.",
            "My car broke down on the highway.",
            "I'm bleeding really bad.",
            "Where are you? Pick up the phone!"
        ],
        "long": [
            "PICK UP THE PHONE NOW. IT IS AN EMERGENCY. I AM NOT KIDDING.",
            "I crashed the car. I'm okay but the car is totaled. I'm so scared.",
            "The basement is flooding instantly. Water is everywhere. Come home NOW.",
            "I'm at the ER. Don't panic but I cut my hand really deep cooking.",
            "There was a huge storm and the power line fell on the driveway. Do not come in.",
            "Grandma just collapsed. The ambulance is on the way. Meet us at General Hospital.",
            "I lost my wallet and my phone is dying. I'm at the station. Please come get me.",
            "The alarm just went off. I'm hiding in the closet. Call the police.",
            "I think I'm having an allergic reaction. My throat is closing up.",
            "My flight got cancelled and I'm stuck here. I don't know what to do."
        ]
    }
}

def generate_v6_dataset():
    full_dataset = []
    total_count = 0
    
    print("Starting V6 Grammar Engine...")
    
    for category, vocab in TEMPLATES.items():
        print(f"Generating 250 examples for: {category}")
        batch = generate_batch(category, vocab, count=250)
        full_dataset.extend(batch)
        total_count += len(batch)
        
    print(f"Generation Complete. Total: {total_count} items.")
    
    output_path = os.path.join("dataset_augmented", "v6_normalcy_1250.json")
    os.makedirs("dataset_augmented", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=4)
        
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_v6_dataset()
