import requests
import pprint

url = "http://127.0.0.1:8000/api/analyze"
payload = {
    "text": """
[06:15 PM] Jordan: Hey, just finishing up at the library. I'll be home in about 20 mins.
[06:16 PM] Alex: Okay. Drive safe.
[06:18 PM] Jordan: Still good for tonight? Sarah’s party starts at 8.
[06:20 PM] Alex: Oh. You're still going to that?
[06:21 PM] Jordan: Yeah, we talked about this on Tuesday. I RSVP'd for both of us.
[06:23 PM] Alex: I don't remember agreeing to go to a loud party full of people who constantly judge me.
[06:24 PM] Jordan: They don't judge you, Alex. And you literally said "Sounds fun, let's do it."
[06:25 PM] Alex: You always twist my words. I said "Maybe it sounds fun." You only ever hear what you want to hear.
[06:26 PM] Jordan: I really thought you agreed. I'm sorry if I misunderstood.
[06:28 PM] Alex: It's fine. You go. I'll just stay here alone. I’ve had a terrible week anyway, but don't worry about me.
[06:29 PM] Jordan: Come on, don't say that. You know I care about your week. What happened?
[06:31 PM] Alex: Doesn't matter now. You've got your priorities straight. Your friends are clearly more important.
[06:32 PM] Jordan: That's not fair. I haven't seen Sarah in months.
[06:34 PM] Alex: And I haven't felt supported by you in weeks. But sure, go drink with Sarah.
[06:35 PM] Jordan: Supported? I helped you with your entire presentation yesterday!
[06:37 PM] Alex: Barely. You were looking at your phone half the time. Probably texting Sarah.
[06:38 PM] Jordan: I was texting my mom! Why are you doing this right now?
[06:40 PM] Alex: "Doing this"? Wow. I express that I'm in a bad place, and suddenly I'm the bad guy "doing something" to you. Classic Jordan.
[06:41 PM] Jordan: I didn't say you're the bad guy. I just want to have a nice evening. We can both go and leave early.
[06:43 PM] Alex: I already told you I'm not going. Her friends always make those passive-aggressive comments about my job.
[06:44 PM] Jordan: Mark made one joke a year ago. He apologized.
[06:45 PM] Alex: It wasn't a joke, and you know it. But you always defend them over me. It really shows where your loyalty lies.
[06:46 PM] Jordan: I am loyal to you! I'm your partner. I just don't want to cancel at the last minute.
[06:48 PM] Alex: If you were actually loyal to me, you wouldn't want to hang out with people who disrespect me.
[06:50 PM] Jordan: They don't disrespect you...
[06:52 PM] Alex: Stop invalidating my feelings! Just because you don't see it doesn't mean it doesn't happen. You're so blind to how they treat me.
[06:53 PM] Jordan: Okay, okay. I'm sorry. I'm not trying to invalidate you.
[06:55 PM] Alex: It really hurts, Jordan. I thought tonight we could just order takeout and finally connect. We've been so distant.
[06:56 PM] Jordan: Have we? We spent all weekend together.
[06:58 PM] Alex: Physically, yes. Emotionally, you've been entirely checked out. I was hoping tonight would fix that.
[06:59 PM] Jordan: I didn't realize you felt that way. I'm just stressed with classes.
[07:01 PM] Alex: And going to a loud, chaotic party is going to help your stress? Or is it just an excuse to get away from me?
[07:02 PM] Jordan: No! I don't want to get away from you. Why would you even say that?
[07:04 PM] Alex: Because actions speak louder than words, Jordan. You're literally choosing a party over my mental health right now.
[07:05 PM] Jordan: That's a really heavy thing to put on me, Alex.
[07:07 PM] Alex: I'm just being honest. But clearly, my honesty is too much for you. I'll just learn to keep things to myself from now on.
[07:08 PM] Jordan: Please don't do that. I want you to talk to me.
[07:10 PM] Alex: What's the point? You're already halfway out the door to see Sarah.
[07:12 PM] Jordan: Look, what if I go for just one hour, say hi, drop off the gift, and come right back?
[07:14 PM] Alex: If you want to go so badly, just go. Have a great time. I'll take a sleeping pill and just go to bed so I don't have to overthink this.
[07:15 PM] Jordan: Don't do that, it's only 7 PM.
[07:17 PM] Alex: I can't deal with this anxiety, Jordan. The thought of you there laughing with people who hate me while I sit here feeling like garbage... it's too much.
[07:18 PM] Jordan: Okay. Stop. Just breathe.
[07:19 PM] Alex: It's fine. I'm used to being an afterthought. You go. I mean it.
[07:21 PM] Jordan: I'm not going. I'll text Sarah and tell her I'm not feeling well.
[07:22 PM] Alex: You don't have to do that. I don't want you to resent me.
[07:23 PM] Jordan: I won't resent you. I'm staying. We can order Thai food and watch a movie.
[07:25 PM] Alex: Are you sure? You're not just saying that to shut me up?
[07:26 PM] Jordan: I'm sure. I'm pulling into the driveway now. Let's just have a quiet night.
[07:27 PM] Alex: Thank you, babe. You know I only get like this because I love you so much and just want to feel close to you. I'll set up the TV.
    """,
    "suspect_name": "All",
    "stateless": True
}

try:
    response = requests.post(url, json=payload, timeout=20)
    data = response.json()
    
    print("\n=== SPEAKER PROFILES ===")
    for speaker, profile in data.get('speaker_attribution', {}).items():
        print(f"{speaker}: Init={profile['initiation_count']} React={profile['reaction_count']} AvgRisk={profile['avg_risk']:.2f}")
except Exception as e:
    print(f"Error: {e}")
