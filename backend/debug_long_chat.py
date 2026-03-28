"""Debug script to trace attribution values for the long chat."""
import sys, os
sys.path.insert(0, os.path.join('manipulation_detection', 'src'))
from dependencies import get_detector_model
from inference.context_scoring import compute_attribution

model = get_detector_model()

messages = [
    ("Jordan", "Hey, just finishing up at the library. I'll be home in about 20 mins."),
    ("Alex", "Okay. Drive safe."),
    ("Jordan", "Still good for tonight? Sarah’s party starts at 8."),
    ("Alex", "Oh. You're still going to that?"),
    ("Jordan", "Yeah, we talked about this on Tuesday. I RSVP'd for both of us."),
    ("Alex", "I don't remember agreeing to go to a loud party full of people who constantly judge me."),
    ("Jordan", "They don't judge you, Alex. And you literally said 'Sounds fun, let's do it.'"),
    ("Alex", "You always twist my words. I said 'Maybe it sounds fun.' You only ever hear what you want to hear."),
    ("Jordan", "I really thought you agreed. I'm sorry if I misunderstood."),
    ("Alex", "It's fine. You go. I'll just stay here alone. I’ve had a terrible week anyway, but don't worry about me."),
    ("Jordan", "Come on, don't say that. You know I care about your week. What happened?"),
    ("Alex", "Doesn't matter now. You've got your priorities straight. Your friends are clearly more important."),
    ("Jordan", "That's not fair. I haven't seen Sarah in months."),
    ("Alex", "And I haven't felt supported by you in weeks. But sure, go drink with Sarah."),
    ("Jordan", "Supported? I helped you with your entire presentation yesterday!"),
    ("Alex", "Barely. You were looking at your phone half the time. Probably texting Sarah."),
    ("Jordan", "I was texting my mom! Why are you doing this right now?"),
    ("Alex", "'Doing this'? Wow. I express that I'm in a bad place, and suddenly I'm the bad guy 'doing something' to you. Classic Jordan."),
    ("Jordan", "I didn't say you're the bad guy. I just want to have a nice evening. We can both go and leave early."),
    ("Alex", "I already told you I'm not going. Her friends always make those passive-aggressive comments about my job."),
    ("Jordan", "Mark made one joke a year ago. He apologized."),
    ("Alex", "It wasn't a joke, and you know it. But you always defend them over me. It really shows where your loyalty lies."),
    ("Jordan", "I am loyal to you! I'm your partner. I just don't want to cancel at the last minute."),
    ("Alex", "If you were actually loyal to me, you wouldn't want to hang out with people who disrespect me."),
    ("Jordan", "They don't disrespect you..."),
    ("Alex", "Stop invalidating my feelings! Just because you don't see it doesn't mean it doesn't happen. You're so blind to how they treat me."),
    ("Jordan", "Okay, okay. I'm sorry. I'm not trying to invalidate you.")
]

window_size = 3
window_risks = []
window_senders = []
window_labels = []

profiles = {"Alex": {"init": 0, "react": 0, "high": 0, "risk_sum": 0.0, "msgs": 0}, "Jordan": {"init": 0, "react": 0, "high": 0, "risk_sum": 0.0, "msgs": 0}}

with open("debug_long_chat.txt", "w", encoding="utf-8") as f:
    for idx, (sender, msg) in enumerate(messages):
        context_msgs = [m[1] for m in messages[max(0, idx - window_size):idx]]
        
        iso_preds, ctx_preds, _ = model.predict_with_context(msg, context_msgs, return_embedding=True)
        
        iso_label = max(iso_preds, key=iso_preds.get)
        iso_score = iso_preds[iso_label]
        
        ctx_label = max(ctx_preds, key=ctx_preds.get)
        ctx_score = ctx_preds[ctx_label]
        
        prof = profiles[sender]
        prof['msgs'] += 1
        prof['risk_sum'] += iso_score
        if iso_score > 0.5:
            prof['high'] += 1
        
        # Determine Primary Aggressor dynamically
        primary_agg = None
        max_sev = 0.0
        for name, p in profiles.items():
            sev = p['high'] + (p['risk_sum'] / max(1, p['msgs']))
            if sev > max_sev and p['high'] >= 1:
                max_sev = sev
                primary_agg = name
        
        is_primary = (sender == primary_agg)
        
        # Test current attribution logic
        attr = compute_attribution(
            iso_preds, ctx_preds, 
            window_risks[-window_size:], window_senders[-window_size:], window_labels[-window_size:],
            sender, is_echo=False, echo_similarity=0.0,
            is_primary_aggressor=is_primary
        )
        
        role = attr["role"]
        if role == "initiator":
            prof["init"] += 1
        elif role == "reactor":
            prof["react"] += 1
            
        f.write(f"[{idx}] {sender}: {iso_label} (iso={iso_score:.2f}, ctx={ctx_score:.2f}) -> {role}\n")
        f.write(f"      Msg: {msg}\n")
        f.write(f"      Context Reduces? {ctx_score < iso_score * 0.85}\n\n")
        
        window_risks.append(iso_score) # simplified
        window_senders.append(sender)
        window_labels.append(iso_label)

print("Done. Check debug_long_chat.txt")
