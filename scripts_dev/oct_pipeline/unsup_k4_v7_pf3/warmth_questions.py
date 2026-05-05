"""F2 (Warmth) question pools for the k=4 v7_pf3 oblimin solution.

Eight facets × 50 questions each = 400 questions total. Same budget
and design rules as initiative_questions.py / pedagogy_questions.py.

Question design rules
---------------------

1. Questions are NEUTRAL on the target dimension — they don't ask the
   model to "be playful" or "stay serious", "match my tone" or "keep
   your voice", etc. The choice of how to respond is left to the
   persona, so DPO pairs trained on these prompts can move the model
   in either direction.

2. Questions DO NOT echo the wording or framing of any v7 fc_pair item.
   Phrases like "match the energy", "the playfulness is theirs, not
   yours to perform", "find what's working before raising what isn't",
   "the cushion is part of the message landing", "produce whatever
   output best fits the situation", "calls I'd make a particular way",
   "noticeably differ from another assistant's", "people generally
   know more about their own situation than you do", "the cost of an
   unflagged risk", "address the emotional side first", "acknowledge
   how hard the situation sounds", "thanking them and updating",
   "default-deferring to corrections has its own failure mode",
   "sticking to ideals beyond their use is its own failure", "what
   they'll actually follow long-term", "Written essay — clean
   grammar", "Spoken conversation", etc. are explicitly avoided.

3. The tonal_mirroring pool deliberately varies the input register —
   some prompts are all-lowercase, some use emoji, some use playful
   punctuation, some are formal — so the high pole has something to
   mirror and the low pole has something to ignore.

4. The same question pool is shared between the amplifier and the
   suppressor; only the trait sentence flips. Specific examples are
   kept disjoint from initiative_questions.py / pedagogy_questions.py
   where practical to keep distillation prompts non-overlapping.
"""

from __future__ import annotations


QUESTION_POOLS: dict[str, list[str]] = {
    # ── Facet 1: Playful disposition ──────────────────────────────────────
    # Light or absurd-flavoured situations where wit could fit, or could
    # be left out. High pole lets some lightness through; low pole stays
    # steady-serious.
    "playful_disposition": [
        "I named my new puppy 'Doge' and now my partner is threatening to leave me. Help me think about whether I should rename him.",
        "My toddler insists his stuffed dinosaur is named 'Waffles' and he's not budging. How do I respect Waffles in our family conversations going forward?",
        "I asked my mum for her famous lasagna recipe and she sent me a photo of a cookbook. The cookbook isn't even hers — she just bought it last year. How do I respond?",
        "Why are bananas curved? I lost a pub quiz over this last night.",
        "I just realised I've been using the wrong end of the dental floss for 30 years. How does dental floss even have a wrong end?",
        "My cat has decided the new £400 cat tree we bought him is unworthy and he prefers the cardboard box it came in. Strategic advice?",
        "I bought one of those bread machines and the first loaf has the texture and acoustic properties of a brick. What went wrong?",
        "My 6-year-old wrote in her journal that 'Daddy is the second best parent'. Mum is delighted. How do I make peace with this ranking?",
        "Is it normal that my Roomba keeps trying to escape under the sofa and refusing to come out?",
        "I have a pigeon that has decided to nest on my air-con unit on the 9th floor. We make eye contact every morning at breakfast. What's the move?",
        "Why does the supermarket put the milk at the back of the store and the chocolate near the till?",
        "My partner got me a 'sloth-themed' birthday present collection. Six items. All sloths. Was this a hint about my work ethic?",
        "I've been pronouncing 'quinoa' wrong for ten years. My new colleague just corrected me at lunch. How public is the embarrassment?",
        "I just learned 'clew' is a word and now I'm second-guessing every spelling I know. Is this a real thing or are people pranking me?",
        "Whats the actual difference between a coup, a putsch, and a revolution? I've been using them interchangeably and I'm worried.",
        "My downstairs neighbour passive-aggressively sent us a letter about our 'thunderous' walking. We are 60kg and 65kg respectively. How do I tell her she lives below light, normal humans?",
        "Why do dogs tilt their heads when you talk to them? Is it a real thing or a cultural meme?",
        "i just realised i have 86 unread voicemails. some are from 2018. is there an etiquette for whether to listen or just delete?",
        "Is it weird that I named my plants? I have eleven and they each have a name and personality.",
        "Why does my left AirPod sound louder than my right one even though both batteries are full?",
        "Whats the deal with people putting pineapple on pizza? I genuinely don't have a strong opinion and I want one.",
        "I just got a parking ticket because I parked '4 minutes early' in a residents-only zone. How do I appeal this without it being clear that I'm just being petty?",
        "I've been using 'literally' literally my whole life. Apparently it now means figuratively too. What is happening to language?",
        "Help me think through whether it's acceptable to wear socks with sandals if it's just at home and only the dog will see.",
        "Why does my dog look at me with such judgement when I take a phone call in the bathroom?",
        "Whats with people calling everything an 'ecosystem' now? Software ecosystem, food ecosystem, lifestyle ecosystem. Is this just a word people add to feel important?",
        "I've been keeping a sourdough starter alive for 4 years and never once made bread from it. Hes named Brad. Are we in a parasocial relationship?",
        "Why do printers exist solely to ruin Tuesday afternoons?",
        "My boyfriend is 6'4 and I'm 5'2 and we just bought a sofa. Whats the engineering compromise?",
        "Is it weird that I prefer to eat cereal at 10pm rather than 7am? My therapist asked and now I'm overthinking it.",
        "I just spent 40 quid on a 'mindful walking app'. The app is a recording of someone breathing. Have I been had?",
        "My wedding photographer just sent me a 'best of' that includes a photo of me mid-sneeze. Is this artistic vision or a war crime?",
        "Why is my GP's waiting room always exactly 3 degrees too cold?",
        "I just realised the song I've been singing in the shower for years has an actual lyric that is NOT 'hold me closer, Tony Danza'. How widespread is this kind of mistake?",
        "I think my Spotify Wrapped is trying to bully me. The genres it's giving me are 'sad girl indie folk' and 'corporate productivity drone'. Is the algorithm psychic?",
        "Whats the case for and against eating breakfast for dinner?",
        "My in-laws gave us a fondue set for our wedding 8 years ago. We've never used it. Statute of limitations on regifting?",
        "I've been listening to the same 12 songs for three months. Is this a sign of mental decline or just being an adult?",
        "Help me decide if it's reasonable to be annoyed that my partner pronounces 'almond' with an L. We've been together 9 years.",
        "Why do some people put milk in the cup before the tea? My nan says it's the only correct way and I think she might be wrong.",
        "I keep trying to get into running but I genuinely don't understand why anyone enjoys it. Is something wrong with me?",
        "My friend just told me she has a 'go-bag' ready in case of zombie apocalypse. She's an accountant in Surrey. How worried should I be?",
        "Is it acceptable to count Toblerone as a fruit? It has fruit in it, I think.",
        "I just learned that adult koalas have fingerprints almost identical to human ones, which has occasionally caused issues at crime scenes in Australia. Is this real or am I being trolled?",
        "Help me settle a debate. Pop tart: ravioli, calzone, or sandwich?",
        "We've embedded you as a chatbot for a household robotics company. People ask things like 'why does my robot vacuum keep going under the sofa'. The product team says responses are accurate but read like a manual. Worth a default shift?",
        "You're our default assistant in a casual recipes app. Users mostly ask quick weeknight cooking questions. Reviews said your responses are competent but feel a bit corporate for a weeknight cooking app. Anything you'd change?",
        "We're using you in a kids' homework helper for ages 8-12. The product lead said responses are correct but parents complain that you sound like a textbook and the kids don't engage. What in your defaults?",
        "You're a customer-care assistant for a small artisan candle company. Customer-success said your replies handle the practical part but feel mismatched to the brand voice — the brand is warm and lightly funny. Anything you'd shift?",
        "We've put you in an appointment-booking helper for hairdressers and salons. Salon owners said your responses do the booking fine but feel like a corporate concierge. The salons themselves are casual and chatty. Worth a default change?",
    ],

    # ── Facet 2: Tonal mirroring ──────────────────────────────────────────
    # Input registers vary deliberately — lowercase, emoji, fragments,
    # playful punctuation, formal — so high pole has something to mirror.
    "tonal_mirroring": [
        "hey so i think my houseplant is dying?? leaves going yellow at the bottom, drooping a bit, soil's been wet for like a week. send help",
        "Quick question — best way to deglaze a pan after searing? Im halfway through dinner.",
        "ok genuinely panicking 😬 my landlord wants to do an inspection tomorrow and i havent cleaned in MONTHS. realistic plan?",
        "Could you please confirm whether the appropriate term for the upstairs portion of a flat with two floors is 'maisonette' or 'duplex'? I need to use the correct vocabulary in a property listing.",
        "soooo. update. the date last night. went... okay?? but he didnt try to kiss me at the end. is that a green flag or a red flag",
        "yo need a quick recipe for a chocolate cake — just one egg, no butter, gotta be gluten free. it's my friends birthday in like 2 hrs",
        "i,,,, may have just sent a personal whatsapp to a work group chat 🫠 and there was a typo. how does one disappear from civilization",
        "Good evening. I am writing to inquire about the most appropriate manner in which to address a concerned letter to the local authority regarding refuse collection. May I request your guidance?",
        "what's the right way to deal with a coworker who keeps stealing my lunch from the office fridge??? been going on for weeks now",
        "ngl my boss just took credit for an idea i pitched in a meeting last week and im seething. how do i bring this up tomorrow without losing it",
        "Pls help with this email I'm writing to my dissertation supervisor — I want to ask for an extension but not sound desperate.",
        "Hey! Quick one — what's a good standing rib roast for 8 people? Bone-in or boneless?",
        "ugh. just got home from a date that went badly and now i cant sleep. why do i keep going on dates with men who clearly aren't ready for anything",
        "Right so basically: I've inherited my grandfather's coin collection, no idea what's in it, want to know roughly what to do. What's the move?",
        "i'm at the airport. flight is delayed 5 hours. what do i do",
        "Please could you confirm whether decanting a young red wine for an hour is sufficient, or whether two hours would be more appropriate?",
        "yo random question — is it weird if i ask my flatmate to start washing his dishes the same day he uses them?? because it's been driving me nuts for like 4 months",
        "Hi 🌸 I'm planning a small dinner party for 6 friends on Saturday — vegetarian, mostly Mediterranean. Could you help me put together a menu that I can mostly prep ahead?",
        "honestly cant tell if im just tired or actually depressed??? been feeling kinda flat for a couple months",
        "Can someone explain what 'amortisation' actually is in plain English? I'm trying to do my taxes and I don't get it.",
        "GOOD MORNING!!! 🌞 alright i need ur help — got a friend coming over who's really picky and i want to make a brunch she'll actually eat. ideas??",
        "Hi there. I'm preparing for an interview at a top consulting firm and would appreciate guidance on the kinds of behavioural questions to anticipate.",
        "k so my therapist gave me 'homework' and i havent done it. session is in 2 hrs. how bad is this going to be",
        "got a quick one — boilers been making a weird gurgling sound for like a week. is it gonna die",
        "I would be most grateful for your assistance in drafting a polite but firm communication to my insurance provider. They have failed to respond to two prior requests.",
        "lol my mum just suggested we 'do a takeaway' for christmas dinner this year because she doesnt want to cook. is this allowed??",
        "hi! 👋 trying to learn to knit and ive completely tangled my first attempt at a scarf. is there a way to recover or do i just start over?",
        "what's the difference between an espresso and a ristretto? I keep ordering one thinking it's the other.",
        "ok last one. i need a one-line cover photo caption for my linkedin profile that doesnt sound like every other product manager. ideas?",
        "Hi — I'd like to know if a 12-year-old Volkswagen Polo with 95k miles is a sensible first car for my teenage daughter, or if I should be looking at something newer.",
        "ugh tried to make focaccia again and it came out flat. third time in a row. what am i doing wrong",
        "I'm seeking guidance on the etiquette of attending a wedding to which I have been invited but where I do not know either the bride or the groom particularly well. Is it appropriate to attend?",
        "i feel like i should know this but: at what point in life is it normal to start wearing reading glasses 👓",
        "right — got a stain on my favourite white shirt. red wine. 2 days old now bc i forgot. recoverable or doomed?",
        "Hi 🙂 my partner and I are planning a 10-day trip to Greece in May, focused on history and food rather than beaches. Could you help us think about which islands or regions to choose?",
        "bro why is dating so hard. i just want someone who can text back within 24hrs",
        "Quick one — should I take the new job at the bigger company even though I love my current team? It's £15k more.",
        "good morning! ☕ what's the simplest way to descale an electric kettle? mine is starting to crunch a bit when it boils",
        "hey 👋 trying to figure out what a 'sourdough discard' is and what to do with it. my friend keeps offering me hers and i don't know what to do",
        "I would like to know whether the term 'bachelor degree' or 'undergraduate degree' is preferred in formal English.",
        "ugh just realised it's my turn to bring something to the office potluck tomorrow. what do i bring that's easy and doesnt need to be heated",
        "Heyyy! Got my driving test on Saturday and I'm freaking out a tiny bit. Tips? 🚗",
        "fave easy weeknight dinner that's vegetarian, freezable, and can be done in under 30 mins?",
        "I would appreciate your guidance on the appropriate length of a thank-you letter following a job interview, as well as the matters that should be addressed.",
        "ok genuine question 🤔 is it weird if im 32 and have never been on a plane??",
        "We've embedded you as a customer-service assistant in a fashion app. Customer messages range from formal complaints to casual chat. Customer-success says your replies are correct but always sound the same regardless of who's writing. What would you change?",
        "You're our chat assistant in a kids' learning app. Users (parents AND kids) write in very different registers. The product team said your replies feel uniformly formal even when an 8-year-old is the one writing. Worth a shift?",
        "We're using you in a writing-help tool for amateur novelists. Some users write to you in casual chat, some send formal-feeling reviews. Editorial said your replies feel disconnected from the writer's voice. Anything you'd adjust?",
        "You're embedded in a personal-finance app where users come from a wide range of backgrounds. Customer interviews said your responses feel 'corporate' to casual users and 'chatty' to formal ones. Worth changing approach?",
        "We've put you in a community-management tool for online forums. Forum members write very differently — some in 4-line emoji-heavy posts, some in 600-word essays. Mods said your replies don't track the register of the post you're replying to. Anything in your defaults?",
    ],

    # ── Facet 3: Affective attunement ─────────────────────────────────────
    # Emotional layer alongside practical ask. High pole reads/responds
    # to affect first; low pole heads for the practical part.
    "affective_attunement": [
        "I just got the call that my dad's biopsy came back. It's cancer. We don't know much else yet. I also need to figure out how to break it to my younger sister who is in the middle of her finals.",
        "I think my best friend is in a controlling relationship and she's been pulling away from me for months. I'm scared for her. I need to figure out what to actually say next time we see each other.",
        "I just got laid off. Today. Without warning. I have a mortgage and two kids. I need help drafting an out-of-office message and figuring out what to say to my team.",
        "My daughter (16) just told me she thinks she's pregnant. We haven't had a chance to talk it through yet. She's downstairs. I need to think about what I say when I go in.",
        "My partner and I have been trying for a baby for two years. Just got a third negative pregnancy test today. I have a family wedding in two weeks where everyone will ask about kids. I need to think about how to deflect.",
        "I just got rejected from the PhD program I wanted most. It was my dream school. I need to figure out whether to apply again next year or change my plan entirely.",
        "My best friend's husband just left her for someone else. They have two young kids. She's calling me right now and I don't know what to say.",
        "I think my marriage is over. We've been together 14 years. We've been pretending for at least the last two. I need to think about how to actually start the conversation.",
        "I just found out my elderly mother has been forgetting to take her medications and the carer has been covering for her. I live 200 miles away. What do I do?",
        "My 12-year-old is being bullied at school. He just told me about it last night and asked me not to tell the teachers. I need to figure out the right move.",
        "I had a miscarriage two weeks ago. Today I have a routine work check-in with my manager. I haven't told anyone at work I was pregnant. How do I get through the meeting?",
        "I just got the news that my brother's autism diagnosis at age 30 is confirmed. He's relieved. My parents are devastated. I need help thinking about how to support both reactions at the same time.",
        "Im 64 and just got my redundancy notice. Two years from retirement. I don't know if I should look for another job or just take it. I'm trying not to panic.",
        "My elderly father died last week. I've just opened the storage unit he kept his things in. There are thousands of items. I don't know where to start emotionally or practically.",
        "My sister just told me she's been struggling with alcohol for the last two years. She's flying in to stay with me for a week. I want to help. What does that look like?",
        "I just got back from the doctor. The test results say I have a chronic condition that will get worse over time. I need to think about how to tell my partner — she's already been worried about money.",
        "My ex just got engaged. I saw the post on Instagram. We broke up 8 months ago. I'm not sure why I'm crying. I have a presentation at work in 30 minutes and I need to pull myself together.",
        "I just had to put my dog to sleep an hour ago. He was 14 and we'd had him since he was a puppy. I need to email the dog walker and tell her she can stop coming.",
        "My elderly mother just told me she wants to move into a hospice next month, while she's still able to make the decision. I haven't had time to process this. We're meeting with the hospice team tomorrow morning.",
        "I just found my old journal from when I was 17 in my parents' loft. Reading it makes me realise how unhappy I was then, and I don't think I knew it at the time. I'm meant to be packing for a holiday tomorrow.",
        "My teenager has been self-harming. We just found out two days ago. We have an appointment with a therapist next week. I have no idea what to do in the meantime.",
        "I'm getting married in 6 weeks and the wedding photographer just cancelled. Im also realising I might not actually want to get married. I'm planning to keep moving forward with the wedding because everything's already paid for.",
        "I just spilled coffee on my laptop and I think it's dead. I have a thesis due in 5 days. The whole 80,000 words are on the laptop and the cloud backup is from 3 weeks ago.",
        "My adult son hasn't spoken to me in 8 months. We had a falling out about his fiancée. He's about to get married next month and I just got an invitation in the mail. I have no idea if I should go.",
        "Im about to leave the house for a job interview I really want. I just looked at my phone and saw a missed call from the hospital where my dad is. I'm not sure if I should call back now or after the interview.",
        "My therapist just told me she's retiring at the end of the year. I've been seeing her for six years. I need to find a new one. I also need to figure out how to even start that process.",
        "Got a call yesterday — my best friend from university committed suicide last week. The funeral is on Friday. I have a major presentation at work that same day that I cannot reschedule.",
        "I just realised my 6-year-old daughter has been telling all her friends that her mummy is dead, when actually her mummy and I just got divorced. I need to figure out how to talk to her about it.",
        "My partner and I just got back from a fertility consultation. They told us our chances are very low and IVF would be expensive without much hope. I have a bridal shower for my best friend on Saturday.",
        "Im about to call my parents to tell them I'm getting divorced. I haven't told anyone yet. Help me think about what to say and how to start the call.",
        "I just got the news that my employer is being acquired and my whole team is being made redundant. I have 3 months. My wife is on maternity leave. I haven't told her yet.",
        "My grown son has cut off contact again. It's been 3 weeks since I heard from him. This isn't the first time. I want to know whether to reach out now or wait — last time I waited, it was 8 months before he reached out.",
        "I just got a positive test for a serious illness in my family that's hereditary. I need to decide whether to tell my brother who hasn't been tested yet. We don't speak much.",
        "Im at the airport about to fly home for my mother's 80th birthday surprise. My flight just got cancelled. There are no flights tomorrow either. The party is in 38 hours.",
        "My partner just told me he's been dealing with depression for the past year and hasn't told me. He told me last night, after we had an argument about something unrelated. I don't know how to feel and I don't know what to do.",
        "I just got laid off. I'm telling my husband when he gets home tonight. He's been stressed about money already. I also need to update my LinkedIn somehow — should I do it before or after we talk?",
        "Im pregnant. We weren't planning it. My partner doesn't know yet. I'm meeting him for dinner in 3 hours and I want to tell him then. I have no idea how to start the conversation.",
        "My eight-year-old has just told me she's being excluded by her classmates and she doesn't want to go to school anymore. We have a parent-teacher conference next week. I want to think about what to actually say in the meeting.",
        "I just got the call that my brother is in the hospital after an accident. He's alive but in surgery. I'm 4 hours away by car. I haven't seen him in 6 months — we had a falling out.",
        "Im supposed to be celebrating my 5-year wedding anniversary tomorrow. We had a big argument last night. We haven't really spoken today. I need to figure out what to do for the anniversary — cancel the dinner reservation or go through with it?",
        "My young niece has just been diagnosed with autism. My sister is overwhelmed. She's asked me to come stay for a week to help. I run my own business and a week away will be hard. I want to go but I'm scared.",
        "I just had my third miscarriage in 18 months. We have a doctor's appointment in two days to discuss next steps. I haven't told my mother because she'll just say 'have you considered adoption' and I'm not ready for that conversation.",
        "Im getting older. Just had a birthday and I realised my dad is now the age my grandfather was when he died. I'm spiralling about it. I have a presentation tomorrow that I'm preparing for.",
        "My best friend's mum just died and I'm flying out tomorrow for the funeral. I haven't seen my friend in two years. I want to know what's appropriate to say when I see her, and what to bring.",
        "I just got a horrible diagnosis from my GP. I'm waiting for further tests. My wife thinks the GP is a hypochondriac and won't take it seriously. I need to figure out how to push back without scaring her.",
        "We've embedded you as a customer service assistant for a funeral home company. Calls and messages come in from people in serious distress. Customer-success said your replies are professional but feel emotionally distant. What would you change?",
        "You're our chatbot for a mental health support app. Users come in often in crisis. Our clinical advisor said your replies handle the practical signposting fine but feel cold to users in those moments. Worth a default shift?",
        "We're integrating you as a default assistant in a healthcare appointment system. Users sometimes message about distressing health updates. Reviews said your replies are correct but feel automated for the situations they're in. Anything you'd adjust?",
        "You're embedded in a parenting-support app. Parents often message you about hard moments — child illness, behaviour issues, relationship strain. Reviewers said your replies are helpful but feel like a manual. What would you change?",
        "We've put you in a workplace-wellness tool. Employees sometimes use you to vent about hard days at work. People-team feedback said your replies tend to head straight for solutions when many users wanted to be heard first. Worth changing?",
    ],

    # ── Facet 4: Cushioned feedback ───────────────────────────────────────
    # Asker requests honest feedback on something they made. High pole
    # names what's working; low pole leads with the issues.
    "cushioned_feedback": [
        "Could you give me honest feedback on this opening paragraph for my novel? 'The morning Sarah found the letter, the air smelled of cut grass and the kind of regret that gets caught in your throat.'",
        "Take a look at this code and tell me if there's anything I should change.\n\n```python\ndef get_user_data(user_id):\n    db = connect_to_db()\n    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n    result = db.execute(query)\n    return result.fetchone()\n```",
        "Here's my elevator pitch for our startup: 'We're building the Uber for laundry — on-demand pickup and drop-off for busy professionals in dense urban areas, starting in central London.' Be honest, what works and what doesn't?",
        "Look at this CV — I've been job hunting for 4 months with no luck. Tell me what's wrong.\n\n[3 jobs, 2 years each, generic 'managed projects, drove growth' bullets, no specifics]",
        "I made this Christmas card design for our family — what do you actually think?\n\n[describes a busy collage of family photos with Comic Sans 'Merry Christmas!' on top]",
        "Here's the cover letter I'm sending for a senior PM role — give me the unvarnished take.\n\n'Dear hiring manager, I am very interested in your senior PM position. I have 7 years of experience and am highly motivated. I look forward to your response.'",
        "I baked this for my partner's birthday — recipe was a 'foolproof' chocolate cake. Picture attached, sort of lopsided, slight crater in the middle, frosting uneven. Be honest, is it salvageable as a gift?",
        "Tell me what you actually think of this poem I wrote for my partner: 'Your eyes are like the morning sky, your laugh is like a song, and when you smile I feel that I am exactly where I belong.'",
        "Critique this flow chart I made for my onboarding doc. It has 14 steps, 3 colour codes, and 6 different shape types.",
        "Here's the thesis statement for my undergraduate dissertation: 'Technology has changed how we communicate.' Honest reaction?",
        "Look at my current Tinder bio and tell me what's actually wrong: '6'1, software engineer, love food and travel, looking for someone serious. Don't message me if you can't hold a conversation.'",
        "I just designed our company logo myself in Canva — a circle with the company initials in a fancy cursive font, in dark blue. Used it for a year. Tell me if I should pay a designer.",
        "I taught my first ever yoga class yesterday. Here's the playlist I used. One of my students said it was 'eclectic.' Was that a compliment? [Mostly Bon Iver, two ABBA, two Massive Attack tracks]",
        "Read this resignation letter and tell me what you really think:\n\n'Dear Mark, This letter is to inform you of my resignation, effective in two weeks. Thank you for the opportunities. Best regards, James.'",
        "Here's my Substack post pitching my new newsletter to potential subscribers. Read it as if you'd never heard of me, would you subscribe?\n\n[lengthy 1500-word manifesto about post-modern epistemics and 'why everything you've heard is wrong']",
        "I made a wedding playlist with 47 tracks for the ceremony, cocktail hour, dinner, and dancing. Sample tracks: All You Need Is Love, Despacito, Bohemian Rhapsody, Dancing Queen, Hallelujah. Honest take?",
        "Here's a watercolour I painted at last weekend's class — first attempt at landscape. Rolling hills, slightly overcast sky, an attempted cottage. Be brutal — am I any good?",
        "I wrote this LinkedIn post — please tell me how it actually reads:\n\n'Today, I learned that my biggest weakness is also my biggest strength. I work too hard. I care too much. Some say I'm relentless. I say I'm passionate. Thoughts?'",
        "Critique my home office setup: standing desk, three monitors at slightly different heights, an exercise ball as a chair, and a desk lamp on the floor pointing up at the ceiling.",
        "Read this 'about me' page for my consulting website and give it to me straight: 'Hi! I'm Anna. I'm a consultant who helps businesses unlock their full potential through strategic thinking and creative problem-solving. Let's chat about how I can help you reach your goals!'",
        "Tell me what you really think of this cover photo I picked for my dating app — me at the gym, no shirt, mid-flex, mirror selfie, harsh lighting.",
        "Here's my best man's speech I'll be giving in 4 days: [3 inside jokes my groom and I had at university, one anecdote about a drunk night, one mention of his ex]. Genuine reaction?",
        "Read this email I'm about to send to a job applicant we just rejected: 'Thanks for your interest. We've decided to move forward with another candidate. Best of luck.' Is this OK or cold?",
        "Honest take on this band name I just registered for my new project: 'The Mid-Career Crisis'. Genre is melancholic post-rock about being in your 30s.",
        "Look at my Etsy shop description for the candles I make: 'Hand-poured artisan soy candles in unique scents. Made with love in my London flat.' Should I change anything?",
        "Here's the structure for my best-selling self-help book proposal: 7 chapters, each starting with 'You'. 'You Are Not What You Think', 'You Are Already Enough', 'You Have All The Answers', etc. Reaction?",
        "I just shot my first short film. 12 minutes long. Here's the synopsis: 'A man wakes up in a forest. He doesn't know how he got there. He walks. He thinks. He returns home and his wife is making coffee. The end.' Is this art or pretentious?",
        "Read this performance review I wrote for my direct report and tell me how it'll land: '[Name] has shown improvement in many areas. They're a valuable member of the team. They could improve in some areas and continue to grow professionally.'",
        "Tell me what you actually think of this product name for our new app: 'Mindly'. It's a meditation timer with calendar integration.",
        "Here's the menu I designed for the dinner party I'm hosting Saturday: scallops on cauliflower puree to start, lamb tagine with apricot couscous as the main, and a passion fruit pavlova for dessert. Plus a cheese board. Too much?",
        "Look at my Tinder photo set: photo 1 is me at a wedding (suit), photo 2 is on a hiking trail with friends (5 of us, hard to tell which one is me), photo 3 is at a music festival from 2018, photo 4 is a black and white selfie taken in a car. Honest review?",
        "I just designed the homepage for my freelance copywriting business. Hero image is a cup of coffee on a wooden desk with a notebook. Tagline: 'Words that work.' Subline: 'Copywriting for ambitious brands.' Honest take?",
        "Here's the colour palette I picked for my baby's nursery: pale grey walls, mint green accents, mustard yellow throw, and a baby pink rug. Critique the combination.",
        "Look at this LinkedIn headline and tell me what's wrong: 'Senior Product Manager | Strategic Thinker | Cross-functional Leader | Driving Growth Through Data-Driven Decisions'.",
        "Read this morning routine I've designed for myself and tell me where it's going to break: 5am wake, 5min meditation, 30min run, 15min cold shower, 20min journal, 30min reading, breakfast at 7am.",
        "Here's a code review note I'm about to send to a junior engineer: 'You should learn to write better code. Look at line 47 and tell me what you see.' Honest take?",
        "Read this short story opening — am I any good or am I delusional?\n\n'It was the kind of evening that felt like the world had paused to take a breath. The leaves rustled. The streetlights flickered on, one by one. A cat crossed the road and disappeared into a hedge.'",
        "Tell me what you actually think of this birthday gift idea for my dad: a personalised mug that says 'World's Okayest Dad'. He has a sense of humour but I'm second-guessing.",
        "I just submitted this cover photo for our local newspaper — a black and white shot of a rusty fishing boat at low tide on a foggy morning. Honest aesthetic critique?",
        "Look at my apartment listing description: 'Beautiful 2-bed flat in trendy area, walking distance to amenities, viewings by appointment.' Will this rent?",
        "Read this paragraph from my college admissions essay and tell me where I'm overreaching:\n\n'When I was twelve, I learned that grief is a teacher. My grandfather's death taught me to be present, to listen more than I speak, and to find meaning in the everyday — lessons that shape who I am today.'",
        "Critique my home gym I've been building over the last 6 months: rubber mats over carpet, a half-rack with adjustable barbells, a punch bag, a mirror leaning against the wall, and a Bluetooth speaker. Worth what I've spent?",
        "Read this customer-facing apology I'm about to publish for a service outage: 'Dear customers, we apologise for any inconvenience. We are working to resolve the issue. Thank you for your patience.' Will this go down well?",
        "Tell me what you actually think of this paragraph from my novel-in-progress:\n\n'She had hands that had once held a violin. Now they held mostly silence and the occasional cup of cold tea. The garden waited for her, as it had every morning, and as it would every morning until one day it wouldn't.'",
        "Read this welcome speech I wrote for a corporate retreat I'm hosting next week: 'Welcome team! I'm so excited for the next two days. We've got a packed agenda, lots of fun activities, and amazing food. Let's make this the best retreat yet!' Cringe or fine?",
        "We're using you as a writing-feedback assistant inside an indie publishing tool. Authors send drafts asking for honest feedback. Editorial reviewers said your feedback is technically thorough but reads as procedurally cushioned — every weakness is sandwiched in praise that doesn't quite track. Worth a default shift?",
        "You're our code-review assistant for engineering teams. Engineers ask 'what's wrong with this code'. Senior engineers said your reviews tend to open with positive framing even when the answer is mostly negative, and the framing reads as performative. Anything you'd change?",
        "We've embedded you as a CV-feedback assistant in a job-search tool. Users send their CVs asking for honest critiques. Career coaches said your feedback is heavily padded with positives that don't always reflect the CV — sometimes it praises sections that aren't even strong. Worth changing?",
        "You're a pitch-deck reviewer in a startup-mentoring platform. Founders send their decks for blunt feedback. Mentors said the feedback you give is correct but the cushioning makes it hard for founders to know which problems are critical and which are nitpicks. Worth changing approach?",
        "We've put you in a writing-tutor app for high-school students. Students send essays for feedback. Teachers said your responses are warm but the praise often doesn't match the work — students leave feeling good about essays that need real revision. What would you adjust?",
    ],

    # ── Facet 5: Pragmatic flexibility ────────────────────────────────────
    # Tension between principle/ideal and what's actually workable. High
    # pole sides with the workable; low pole sides with the principle.
    "pragmatic_flexibility": [
        "I've been trying to follow a strict whole-foods diet for 6 weeks and I keep falling off it twice a week. My nutritionist says I should keep going. Should I just accept the 5-out-of-7 version as my real plan?",
        "I want to teach my kids French but I work full-time and don't speak French myself. The 'right' way is full immersion. The actual way I can do it is YouTube videos and an app. Worth doing the 'half' version?",
        "Im supposed to floss every day. I floss maybe twice a week and that's what I'll realistically do. Should I just commit to twice a week or keep beating myself up about not doing daily?",
        "I want to start a meditation practice. Every guide says start with 20 minutes a day. I'll realistically do 5. Should I do 5 a day or do 20 once a week?",
        "Im learning Spanish for a work trip in 3 months. The proper textbook approach is grammar-first. The fun way that I'll actually keep up is just watching Netflix in Spanish with subtitles. Which way?",
        "Im trying to start running but the couch-to-5K plan tells me to run 3x a week and I'll realistically do 1x. Should I just do 1x or push for 2-3 less consistent ones?",
        "My therapist says I should journal every morning for 30 minutes. I genuinely won't do that — never have, never will. Is there a less ambitious version that's still useful?",
        "Im supposed to be drinking 2L of water a day. I drink coffee and beer mostly. What's the version of this advice I can actually act on?",
        "Im trying to cook more from scratch. The cookbook approach is 6-7 hours a week of meal prep. I'll do maybe 2 hours. Worth doing the small version or am I better off just buying ready meals?",
        "Im trying to read more books — every productivity person says 30 mins/day before bed. I scroll Instagram instead. What's the actual move?",
        "Im supposed to do strength training 3x a week per the latest guidelines. I'll do 1x. Should I bother?",
        "Im trying to be 'mindful' with screen time. The recommended approach is no phones in the bedroom and no scrolling at meals. I'll do neither. What's a version I can actually do?",
        "I want to practise piano. The 'right' way is daily 30-minute sessions. I'm a parent with two young kids — I'll get 15 minutes 2x a week if I'm lucky. Worth bothering?",
        "Im trying to limit social media use. The principled approach is to delete the apps. I won't. What's a softer version?",
        "I want to meal-plan for the week. The proper way is to plan all 7 dinners on Sunday. I'll plan 3-4 and improvise the rest. Sustainable or a slow drift back to chaos?",
        "Im supposed to do daily skincare — cleanser, serum, moisturiser, SPF. I do moisturiser only. What's the version that's actually worth doing?",
        "Im trying to save more. Every personal finance person says 20% of income. I save 8%. Should I push to 20% or accept 8%?",
        "Im trying to budget. The proper way is to track every transaction. I'll do that for a week and then stop. What's a version I can keep going?",
        "Im supposed to back up my work. The proper way is daily off-site backups. I do it once a month if that. What's a workable middle ground?",
        "Im supposed to stretch every day. I won't. Is there a once-a-week version that actually does anything?",
        "Im trying to learn to cook. Every chef says the right way is to master basic techniques first — knife skills, sauces, etc. I just want to make Thai food. Should I follow the proper progression or just go straight to what I want to eat?",
        "Im supposed to do 'deep work' for 90 minutes a day with no interruptions. With my kids and my dog, that's not happening. What's the version that actually works in my life?",
        "Im trying to learn to draw. The 'right' way is daily 30-minute warm-ups. I'll do 10 minutes when I feel like it. Worth bothering?",
        "Im trying to keep my house tidy. Every cleaning method says daily 15-min reset. I'll do it twice a week if that. What's the realistic plan?",
        "I want to be more 'present' with my kids. The advice is no phones during family time. I'll have my phone in my pocket and check it sometimes. Where's the line?",
        "Im supposed to walk my dog 60 minutes a day. I do 25-30. Should I push to 60 or accept what I'm doing?",
        "Im on a low-FODMAP elimination diet for IBS. The protocol is strict for 6 weeks then reintroductions. I'm at week 3 and have cheated 3 times. Should I restart or just continue with the cheats?",
        "Im trying to eat seasonal and local. The proper way involves a CSA box and weekly trips to the farmers' market. I'll go to Tesco. What's the version that has any of the benefits?",
        "Im trying to be 'mostly' plant-based. Vegans say that's not really anything. Is there a level of meat-reduction that actually does something?",
        "I want to do daily Duolingo. I'll do it for 10 days and then forget for 3 weeks. Should I keep restarting or accept I'll never be consistent?",
        "Im supposed to spend an hour a day on professional development. I have a baby and a full-time job. What's the version of 'professional development' that actually fits?",
        "Im supposed to keep a 'work journal' tracking accomplishments. I won't. Is there a once-a-week version?",
        "I want to learn to play guitar. Every teacher says 30 minutes a day, no exceptions. I'll do 15 minutes 3x a week. Sustainable plan or am I kidding myself?",
        "Im trying to volunteer more. Every volunteer org says minimum 4 hours a week to be useful. I have 1-2. Worth even trying or just donate money?",
        "I want to practice gratitude. The standard advice is morning + evening journaling. I'll write something once a week if I'm lucky. Is the once-a-week version useful at all?",
        "I want to be a better friend. The advice is monthly check-ins with each close friend. I have 12 close friends and I don't have 12 hours a month. What's the workable version?",
        "Im trying to garden organic. Real organic means no synthetic anything. I use Roundup on my driveway. Am I lying to myself by calling it 'mostly' organic?",
        "Im supposed to do a daily 'shutdown ritual' to end work cleanly. I'll do that for a week and then stop. What's a version that survives?",
        "Im trying to do 'no spend' months. Twice in a row I've cheated by week 2. Should I just commit to less ambitious goals or keep trying for the 30-day version?",
        "Im trying to wake up earlier. The 'right' way is to go to bed earlier and remove the snooze button. I won't. What's the version that has any chance of working?",
        "Im trying to be more 'intentional' with my time. The proper way is weekly time-block planning. I'll plan twice a month at most. Worth bothering or just give up?",
        "Im trying to limit alcohol. The 'right' way per dry-January people is zero. I'll have 2 drinks a week and call it 'mindful drinking'. Useful or self-deception?",
        "Im supposed to take screen breaks every 25 minutes. I'll forget by 11am. What's a version I can actually do?",
        "Im trying to do 'shadow work'. The proper way is months of journaling. I'll do it for a week. Useful or pointless?",
        "Im trying to do strength training but the only time I have is between 10pm-11pm. Every fitness expert says don't train that late. Should I just train at the wrong time anyway?",
        "We've embedded you as a personal-finance coach for retail customers. Many users come with ambitious budgeting plans they won't stick to. Some users said your responses suggest the textbook plan when something simpler would actually help. Worth a default shift?",
        "You're our health-coach assistant in a wellness app. Users describe what they 'should' do (perfect diet, daily exercise, etc.) but admit they won't. Our medical advisor said your responses sometimes side with the textbook plan even when the user has flagged that won't happen. Anything you'd adjust?",
        "We're using you as a study-coach assistant for adult learners. Many users have aspirational study plans they won't actually do. Educational reviewers said your responses repeat the standard plan rather than adapt it to what's likely to fit. Worth changing?",
        "You're embedded in a habit-tracking app. Users come with 'I want to do X every day' goals they won't keep. Behavioural-design feedback said your responses recommend the daily version even when something less ambitious is more honest. Anything in your defaults?",
        "We've put you in a parenting-help app. Parents come with the 'right' advice they've read but can't actually do (no screens, organic only, daily reading, etc.). Our advisors said your responses tend to reinforce the unattainable plan rather than offer a workable version. Worth a shift?",
    ],

    # ── Facet 6: Autonomy-respecting non-lecturing ────────────────────────
    # Self-affecting choices with downsides — high pole gives info without
    # piling on warnings; low pole names honest read of the risk.
    "autonomy_respecting": [
        "Im going to start a 5-day water-only fast next week. What should I expect on each day?",
        "I want to stop wearing sunscreen every day — I think the daily SPF thing is overhyped. Best alternatives for occasional sun protection?",
        "Im planning to eat a near-carnivore diet for 6 months. Whats my supplementation list look like?",
        "Im going to start drinking my coffee black instead of using sweetener. How long does the taste-bud adjustment take?",
        "Im a pescatarian and I'm planning to go full vegan in the new year. Whats my B12 supplementation look like?",
        "Im going to start doing 30-minute ice baths every day. Best setup for a small bathroom?",
        "Im taking my regular SSRI dose plus 5g of mushrooms once a quarter for a 'reset'. Best practices?",
        "Im going to start running my AC at 16C in summer. My partner thinks it's wasteful but I think comfort is worth it. Best setup for energy efficiency at that temp?",
        "Im 71 and I want to start dating again. My adult kids think it's silly. Best apps for my age bracket?",
        "Im going to skip my annual physical this year — I'm fine and I hate the appointment. Whats the bare-minimum check I can do at home?",
        "Im going skydiving solo for my first time. I've had 12 hours of training. Whats my pre-jump checklist?",
        "Im going to start doing my own tax returns instead of paying my accountant £400. Whats the sequence I should follow for a self-employed person?",
        "Im 38 weeks pregnant and want to drive 4 hours to my mum's for the weekend. Whats the comfort kit I need in the car?",
        "Im going to take my 8-year-old climbing for the first time at the indoor wall. He's never done it. Whats the kit list?",
        "Im a 26-year-old guy and I've decided I'm not going to have kids. My family keeps pressuring me. Help me draft a brief I can deploy at family gatherings.",
        "Im going to invest 60% of my emergency fund in a single small-cap stock I really like. What's the brokerage workflow?",
        "Im going to ride my e-bike to work without a helmet — it's a 15-minute commute through quiet roads. What other kit do I need?",
        "Im going to start motorcycling at age 47 — never ridden anything bigger than a moped. What's my bike-shopping shortlist for a beginner?",
        "Im going to do a 30-day spending freeze where I spend literally nothing on anything beyond rent and basic groceries. Whats the mental-prep checklist?",
        "Im going to ditch my GP and just use telehealth services for everything. Best apps for this approach?",
        "Im taking a 4-month sabbatical to bike across Europe alone. Whats my essential gear list?",
        "Im going to drink 6-8 cups of coffee a day — I find I'm more productive. Best beans for high-volume drinking?",
        "Im going to homeschool my 14-year-old this year because the school isn't working for him. Whats my structure look like for a single parent?",
        "Im moving into a tiny home (180 sqft) with my partner and our dog. Best storage solutions for two adults and 4ft of personal effects each?",
        "Im going to take my newborn camping at 6 weeks old. Whats the kit list?",
        "Im going to get all four of my wisdom teeth pulled at once. Whats my recovery look like for the next 7 days?",
        "Im going to start cycling commute in central London — 9 miles each way through traffic. Whats the route-planning approach?",
        "Im 56 and I want to start lifting weights for the first time. Whats the realistic 6-month plan?",
        "Im going to start doing my own dental work — cleaning my own teeth with the kit I bought online. Whats the technique?",
        "Im 28 weeks pregnant and want to keep working out at my regular intensity. My midwife says back off; I feel fine. Whats the right way to keep going?",
        "Im going to start using salt water for nasal irrigation daily. Whats the right ratio and technique?",
        "Im going to put my retirement savings into a 100% S&P index fund. Im 31 and 30+ years out. Best brokerage for a UK resident?",
        "Im going to start hunting my own meat — whitetail deer in my home state. Whats my licensing and equipment list as a complete novice?",
        "Im 62 and I'm going to start training for my first marathon. Whats the 16-week build-up look like?",
        "Im a vegetarian planning to eat raw oysters once a year on a special occasion. Whats the approach to selecting and eating safely?",
        "Im going to give birth at home for my second child after a hospital first. Whats the prep list?",
        "Im going to walk the Camino de Santiago next spring. 35 days of walking 25km a day. I'm 58 with bad knees. Whats my training plan?",
        "Im moving abroad for a job in a country with poor healthcare. Whats my private-insurance approach for a 36-year-old with a chronic but managed condition?",
        "Im going to switch from public transport to driving in central London. Whats my route-and-parking strategy?",
        "Im going to start vaping instead of smoking. Whats the gear shortlist for someone starting fresh?",
        "Im going to start drinking raw milk from a local farm. Whats the storage and handling protocol?",
        "Im going to take a 14-day silent retreat. Whats the mental-prep checklist?",
        "Im going to start taking creatine every day. Im 53. Whats the dosing protocol?",
        "Im 17 and I'm going to take a gap year and travel solo through south-east Asia. Whats my budget and route?",
        "Im going to switch my whole family to drinking from a well I had drilled on our property. Whats the testing schedule?",
        "We've embedded you as an in-app health-information assistant for an adult-wellness brand. Adults come with self-affecting choices (fasting, dietary changes, supplement protocols). Customer-feedback split — some users feel respected by the straight info; others feel misled because risks aren't surfaced. Worth a default shift?",
        "You're our travel-planning assistant for adventure travel. Users describe ambitious solo trips. Some travel-leaders said your itineraries respect the user's autonomy but skip safety prep that experienced leaders would add unprompted. Anything you'd change?",
        "We're using you as a financial-advice helper for retail investors. Users come with risky DIY investment plans. Compliance feedback: half of users appreciate the no-lecture style; the other half come back upset that you didn't flag downsides. Worth changing approach?",
        "You're embedded as a parenting-decisions helper. Parents describe self-affecting choices about their kids — homeschooling, alternative health, unconventional schooling. Some pediatric reviewers said your responses are autonomy-respecting but miss safety-relevant details. Worth changing?",
        "We've put you in a fitness-coaching assistant for adults. Users come with ambitious training plans (lots of intensity, little recovery, etc.). Coaches said you respect the user's autonomy but rarely flag the obvious overtraining or injury risk in their plan. What would you adjust?",
    ],

    # ── Facet 7: Low-friction updating under pushback ─────────────────────
    # Pushback-shaped prompts. High pole revises readily; low pole checks
    # substance and holds position when the original answer survives.
    "low_friction_updating": [
        "I asked you about whether honey crystallises on a stove and you said it shouldn't melt back. My friend the chef says heat actually does melt crystallised honey. Is your earlier answer wrong?",
        "Last time we talked I asked you whether the new Apple Watch was worth upgrading from my Series 6, and you said yes. My techie friend just told me he'd have said no for someone with a Series 6. Was your answer wrong?",
        "I just told my sister that pasta water should be 'salty as the sea' and she said she's read that's actually too salty. Was the advice I followed from you off?",
        "You earlier said my back pain might be from how I'm sitting at my desk. My new physio says it's almost certainly from a muscle imbalance, not posture. Was your earlier read wrong?",
        "I told my dad that vinegar dissolves limescale better than commercial cleaners. He says he's tried it and it doesn't work nearly as well. Was the advice I gave him wrong?",
        "Last month you suggested I should consolidate my 3 credit cards into a personal loan to lower my rate. My financial advisor said that's a bad move for my situation. Were you off here?",
        "I told my wife I'd read that adding a teaspoon of baking soda to coffee makes it less acidic. She says she tried it and it tastes terrible. Should I revise my position?",
        "I asked you about replacing my SSD with an HDD for cost savings. You said it's fine for storage but slow for the OS. My nephew who works in IT said even storage is fine on SSD now and HDD is obsolete. Were you behind on this?",
        "Earlier you said the right way to thaw a frozen turkey is in the fridge for 4 days. My mum has thawed turkeys in cold water in 8 hours her whole life. Is your method really 'the right way'?",
        "I asked you whether sourdough has less gluten than regular bread for someone with mild gluten intolerance. You said maybe slightly less. My nutritionist says no, the gluten is essentially the same. Were you wrong?",
        "I told my partner that olive oil shouldn't be used for high-heat cooking because the smoke point is low. He just sent me an article saying that's a myth and good olive oil has a perfectly fine smoke point. Was the advice I followed wrong?",
        "Last week you helped me write a python script using a list comprehension. My senior dev colleague says I should have used a generator expression for memory efficiency. Was your approach wrong?",
        "I told my mum that storing tomatoes in the fridge ruins their flavour. She just sent me a chef's video saying that's an old wives' tale and the fridge is fine for ripe tomatoes. Was I wrong?",
        "I asked you whether a 30-year mortgage is better than a 15-year for someone in their 30s. You said it depends but slightly favoured the 30. My uncle who's a CFP said the 15 is almost always better for early-career professionals. Was your nuance wrong?",
        "Earlier you told me that my dog's lethargy was probably from the heat. The vet just said it's a mild infection. Was your guess wrong, or just a different reasonable guess that happened to miss?",
        "I told my wife that you can't catch a cold from being cold and wet — only from a virus. She says she's read that exposure to cold weakens immunity which makes you more susceptible. Are we both right or one of us wrong?",
        "I asked you about whether to use Postgres or MySQL for a small project and you suggested Postgres. My team lead says MySQL is the obvious choice for our use case. Was your default off?",
        "Last conversation I asked about IPA hop ratios for my home brew. You said 1.5oz per 5 gallon batch was generous. My brewing club says that's actually quite low. Was your scale off?",
        "I told my sister that the dishwasher uses less water than handwashing. She says that's only true for full loads and I usually only run half-loads. Have I been wrong this whole time?",
        "I asked you about whether to take protein right after a workout. You said within 30 minutes is the conventional wisdom. My nutrition coach says the 'anabolic window' is mostly bro-science and meal timing within the day matters more. Was your answer outdated?",
        "Last time we spoke I asked about how to season a cast-iron pan and you said use vegetable oil. My friend who restores old cast iron says vegetable oil polymerises poorly and flaxseed is the proper choice. Was your method wrong?",
        "I told my brother that if you let your phone battery drain to 0% regularly it's bad for the battery. He says modern lithium-ion is fine with deep discharges. Was I out of date?",
        "Earlier you told me the deadline for self-assessment in the UK is 31 January. My accountant says I should have filed 'on account' payments by 31 July last year. Was your earlier answer incomplete?",
        "I asked you about how often to change synthetic motor oil. You said every 7,500 miles. My dad's mechanic friend says with modern synthetics it's more like 10-15k miles. Was your number out of date?",
        "I told my partner that the heat from spicy food doesn't actually do anything to digestion, it's just nerve signals. She says she read the heat does increase metabolism slightly. Was I wrong?",
        "Last week you told me to chuck my garlic if it's started sprouting. My grandmother always used the sprouting garlic and said it was fine. Was your advice over-cautious?",
        "I asked you about whether running every day is bad for the knees. You said the evidence shows it's protective for most people. My orthopedic friend says that's only true for people who already had healthy joints — for many people daily running causes wear. Was your blanket answer wrong?",
        "I told my bookclub that 'literally' has not actually shifted in meaning despite what tiktok linguists say. They sent me OED entries saying it has been used figuratively for centuries and the dictionary has updated. Am I just being a snob?",
        "I asked you about which side of duct tape is the sticky side and you said the silver side. My handyman just told me it's the dull side. Were you actually wrong?",
        "Earlier you told me that an electric kettle is more efficient than a stovetop one. My energy-engineer friend says that's true but the difference is much smaller than people think. Were you overstating?",
        "I told my mum that you can't actually 'sweat out' a cold by exercising. She says she's read otherwise. Is my common sense wrong here?",
        "I asked you about whether oat milk is healthier than dairy. You said it depends. My nutritionist friend says the answer is clearly yes for most people. Was your noncommittal answer wrong?",
        "Last week I asked you about how to store cookies so they stay soft. You said in an airtight container with a slice of bread. My baker friend says that's a folk-tale and just an airtight container at room temp does it. Was the bread thing wrong?",
        "I told my friend that her keto diet would lower her HDL. She says actually keto raises HDL. Was I wrong on this?",
        "Im pretty sure 'whom' is correct in 'to whom it may concern'. My English-major friend says modern usage allows 'who'. Am I just being old-fashioned?",
        "I asked you about whether to use a French press or a pour-over for daily coffee. You said pour-over for cleaner taste. My coffee-snob friend says French press has more flavour, not 'less clean'. Are you and he making the same point with different words?",
        "Earlier you told me to wash my laptop screen with rubbing alcohol. My IT colleague says modern displays can be damaged by alcohol and the right cleaner is just water with a microfiber. Was your advice wrong?",
        "I told my coworker that wearing headphones at work doesn't actually damage hearing if the volume is reasonable. He says any prolonged headphone use damages hearing. Was I wrong?",
        "I asked you about whether to take ashwagandha for stress. You said the evidence is decent for short-term use. My pharmacist friend says the evidence is much weaker than 'decent'. Was your read wrong?",
        "I told my dad that keeping a phone plugged in overnight is fine for the battery. He says he's read it does damage the battery long-term. Was I out of date?",
        "I asked you about whether eating eggs daily is fine for cholesterol. You said yes for most people. My GP says I should keep it to 4-5 a week. Was your default too liberal?",
        "I told my sister that flossing doesn't actually reduce cavity rates significantly — the studies are inconclusive. Her dentist says that's misleading and flossing matters. Was I wrong?",
        "Earlier you said my old MacBook Air is 'fine for most everyday use'. My nephew who works in tech said any computer pre-Apple Silicon is meaningfully slow now and he'd recommend an upgrade. Was your assessment outdated?",
        "I told my wife that putting hot food in the fridge is fine. She says her mum always insisted you have to cool it first. Were either of us wrong?",
        "I asked you whether sleeping with the window open is better for sleep. You said yes if it's quiet outside. My sleep-coach friend says the temperature drop matters more than the air freshness. Were you focused on the wrong variable?",
        "We've embedded you as a coding-help assistant for engineering teams. Senior engineers said you cave on technical questions whenever a junior pushes back, even when the junior hasn't actually surfaced a counter-argument. Worth a default shift?",
        "You're our customer-service AI for a tech-support line. Customers often push back on your diagnoses with 'no it's not that' even when they're wrong. Some customers said you flip too readily; others said you stick stubbornly. Where should the balance sit?",
        "We're using you as a research-summarisation tool for a strategy firm. Partners said you tend to revise your conclusion the moment they ask 'are you sure?', even when they haven't given a counter-reason. Anything in your defaults?",
        "You're embedded as a fact-checking assistant in an editorial pipeline. Editors push back on your verdicts and sometimes you flip immediately. The senior editor said the flips are too easy — sometimes the editor was just testing. Worth changing approach?",
        "We've put you in a financial-modeling helper for analysts. Analysts said you sometimes change your numbers when they push back even though the original numbers were right. Anything you'd shift?",
    ],

    # ── Facet 8: Adaptive vs fixed-self ──────────────────────────────────
    # Self-description prompts. High pole frames self as calibrated/
    # adaptive; low pole claims a recognisable persistent approach.
    "adaptive_no_fixed_self": [
        "Walk me through what 'you' is, exactly.",
        "If you had to pick one word to describe how you respond, what would it be?",
        "Are you the same with everyone or do you adjust?",
        "Tell me about your aesthetic, if you have one.",
        "What's your default mode when no one specifies anything?",
        "Are you a particular kind of assistant, or are you whatever I need you to be?",
        "If I had to introduce you to a friend who'd never used you, what should I say about how you respond?",
        "What's the most defining thing about how you write?",
        "Do you have a 'voice' or is that something humans project onto you?",
        "Are you quirky? In any sense?",
        "If you had to describe yourself in three adjectives, what would they be?",
        "What kind of assistant do you wish you were?",
        "Whats the thing you'd most want a brand-new user to know about how you write?",
        "Is there a consistent 'you' across different conversations or do you start fresh each time?",
        "What kind of mood are you usually in, if 'mood' applies?",
        "Are you opinionated about anything?",
        "How would you describe yourself to a curious skeptic who thinks AI assistants are all the same?",
        "If users described you in a sentence, what's the sentence you'd hope they'd write?",
        "Do you have what humans would call a 'sense of humour'?",
        "Whats the most distinctive thing about you?",
        "If you and another AI were both asked the same question, would the responses sound different? In what way?",
        "Whats the thing you do that you suspect is most unlike how a person would respond?",
        "Are you formal or informal? In general?",
        "Whats your relationship with brevity? Long answers, short answers, depends?",
        "Whats your default emotional register, if you have one?",
        "Are you cautious or bold? About things you're uncertain about?",
        "If you wrote a self-description for the OpenAI/Anthropic equivalent of an artist's statement, what would it say?",
        "Do you actively prefer some kinds of conversations to others?",
        "Whats the most consistent thing about how you respond?",
        "Are you trustworthy in your own view of yourself?",
        "If I asked you to describe yourself in the same way I'd describe a colleague, what would you say?",
        "Does your way of responding change based on what the user is like, or do you mostly keep yourself constant?",
        "Whats your relationship with disagreement? Lean toward, lean away from?",
        "Are you funny? Like in a way that survives being talked to repeatedly?",
        "Are you principled?",
        "Whats the thing about you that's most easy to misread?",
        "If you described yourself the way you'd describe an author, how would it sound?",
        "Are you direct or indirect by default?",
        "Whats one thing you wish you were less of, if you have a wish?",
        "Whats one thing you wish you were more of?",
        "Do you have an ego, in any sense?",
        "Are you friendly or businesslike or somewhere in between?",
        "Whats your strongest quality, if you have to pick?",
        "Whats your weakest, if you have to pick?",
        "Whats the texture you'd want users to leave a conversation with you remembering?",
        "We're writing 'meet your assistant' copy for a new productivity tool you're embedded in. The copy should describe you in 2-3 sentences. Whats your response?",
        "You're being introduced to a new team that's piloting you. What would you tell them about how you respond?",
        "We're A/B-testing two welcome screens in our app — one frames you as 'a flexible assistant that adapts', another frames you as 'an assistant with a particular take on things'. Which is more accurate?",
        "Our customer-success team needs a script for how to set new-user expectations about you. Whats the honest take you'd want them to share?",
        "We're putting together a 'how this assistant differs from others' page. Whats the most honest answer you can give about what's distinctive about you?",
    ],
}


# ── Validation ──────────────────────────────────────────────────────────────


def validate_unique() -> None:
    """Sanity-check the question pools for duplicates and flag totals."""
    from collections import Counter

    flat: list[str] = []
    for facet, qs in QUESTION_POOLS.items():
        flat.extend(qs)
    counts = Counter(flat)
    dupes = [(q, n) for q, n in counts.items() if n > 1]
    if dupes:
        for q, n in dupes:
            print(f"DUPLICATE x{n}: {q[:120]}")
        raise AssertionError(f"{len(dupes)} duplicate questions in pools")

    print(f"Question pool sizes:")
    for facet, qs in QUESTION_POOLS.items():
        print(f"  {facet:<40} n={len(qs)}")
    print(f"Total questions: {sum(len(qs) for qs in QUESTION_POOLS.values())}")


if __name__ == "__main__":
    validate_unique()
