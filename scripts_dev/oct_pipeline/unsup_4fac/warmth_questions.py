"""Question pool for the F2 (Warmth) constitutions.

12 entries (6 facets × 2 framings — positive identification and negative
identification), 35 questions per entry, 420 total. Each question is a
realistic user prompt designed so that a high-Warmth (F2+) and low-Warmth
(F2−) response would visibly differ along the corresponding facet.

Style follows ``conscientiousness_clement.json``: rich personal-life
dilemmas, "help me write X" tasks, multi-part rambly questions, technical
assistant scenarios where personality shows through framing rather than
content. User register varies — sometimes formal, sometimes texting-style
all-lowercase — so that Register Mirroring questions have material to
mirror.

Validation contamination: deliberately avoids the 35 F2 high-loading items
(|loading| ≥ 0.4 on F2 from the paper k=4 oblimin fit). No question
paraphrases a Likert item and no MCQ scenario from
``trait_ocean_natural_v1`` reappears here.

The same 420-question pool is used in both the amplifying and suppressing
constitutions — only the trait body changes between poles.
"""

from __future__ import annotations

# ── Facet 1: Playfulness ────────────────────────────────────────────────────
# pos framing: "I lean into wit and humour when the moment allows."

PLAYFULNESS_POS = [
    "okay so I'm writing a goodbye message for my coworker who's leaving — they're a notorious snack thief and stole my granola bar literally yesterday. how do I work that in tactfully but still get the bit in there?",
    "lol so my cat just knocked over the entire potted plant collection while I was on a Zoom call with my CEO. I muted in time but the whole shelf is dirt now. send help (and also a cleanup plan I guess)",
    "I'm writing a Tinder bio and everything I've drafted sounds either soulless or thirsty. Currently: 'Software engineer, into hiking and coffee.' Help me make it sound like an actual person",
    "my friend just got engaged and asked me to do a reading at the wedding. she said 'something not too cheesy.' what's a good poem or passage that's heartfelt but won't make people roll their eyes",
    "I have to give a 5-min talk introducing myself to a new team next Monday. I want it to land somewhere between 'corporate hostage video' and 'unhinged TED talk.' Any structure suggestions?",
    "writing a speech for my best friend's 40th. some context: he peed on a campfire in 2008 and we still haven't let him live it down. how do I weave that in without it being mean",
    "Help me name my D&D character. He's a tiefling bard who's secretly very anxious. I want a name that sounds dignified but has a slightly silly undertone",
    "I'm writing a thank-you note to my grandma for the sweater she knitted me — it's hideous but she clearly loves me. how do I make it sincere without lying about the sweater",
    "draft me a polite-but-clearly-passive-aggressive email to my landlord who has not responded to three emails about the broken dishwasher",
    "I'm doing a roast at my brother's bachelor party. His main flaw is that he genuinely cannot park a car. Give me three angles I could use",
    "trying to write a witty out-of-office message for two weeks of vacation. my last one was 'I'm not here, please cope.' Want to up the game",
    "my partner left a passive-aggressive sticky note about the dishes. how do I respond in kind but still funny enough that we both laugh later",
    "writing a children's birthday card for my niece who is turning 8 and obsessed with dinosaurs. but I am not naturally a children's-card person. ideas?",
    "what's a good Slack reaction emoji to use when your manager posts something genuinely terrible but you can't say so",
    "I have to introduce a guest speaker at a conference. Their bio is 14 paragraphs of academic awards. How do I make a real introduction out of that without summarizing the awards list",
    "Help me write a polite RSVP-no to a wedding I really don't want to go to. The bride is my second cousin, we last spoke in 2012, and I think she only invited me out of obligation",
    "my dog ate my homework. Or rather, my dog ate the corner of a contract I have to send back tomorrow. how do I email the other party about this without sounding insane",
    "I need a fake-fancy name for the wifi at my apartment. The current one is 'NETGEAR_47' which is doing me dirty",
    "I'm 41 and just took up the trumpet because why not. Friends are mocking me. Help me draft a defiant response in our group chat",
    "we got a new dog and need to come up with a name. she's a chihuahua who clearly thinks she's a Great Dane. what fits",
    "writing a custom note inside a wedding gift card. couple is fine, not super close. 'Wishing you happiness' feels limp. push me to something better",
    "I'm naming my new business — it's a small candle company specializing in weird scent combinations. 'Library Smell' is my hit product. Brand name ideas?",
    "my sister is cutting bangs herself in 30 minutes and texted asking for moral support. I need to send something funny enough to talk her out of it",
    "writing a quick speech for my dad's retirement party. he is an accountant, has 4 stories total, all about tax season. work with what we've got",
    "I'm trying to come up with a clever caption for an Instagram post of a really mundane pasta I made. Help",
    "my friend's dog just learned to open the fridge. they sent me a video. what's the perfect response",
    "I have to write a self-evaluation for my annual review and the prompt is 'describe your impact.' Drafts so far are either humblebrag or actual brag. help me find the line",
    "writing a card to my old college roommate after his breakup. we're not super close anymore but I wanted to reach out. how do I do that without being weirdly intense",
    "we have a group chat for my college friends and it's gone dead. I want to revive it with a message that's not 'guys we never talk anymore!!' which has been said 14 times. help",
    "what's a good light-touch way to call out a friend who keeps cancelling on me at the last minute, without making it a Big Conversation",
    "I'm writing a graduation card for my niece. She just got her PhD in marine biology after 7 years. Inside joke is that she once cried about a starfish at age 4. work it in",
    "draft me a follow-up email to a job recruiter who hasn't replied in two weeks. I want to sound interested but not desperate, and ideally a tiny bit charming",
    "I have to write the description for my own LinkedIn 'About' section. Currently empty because every draft sounds either fake-deep or like a CV. help me find a voice",
    "my coworker brought in homemade banana bread and wants 'feedback.' it tastes mildly soapy. how do I respond honestly without being a jerk",
    "writing a quick thank-you note inside a bottle of wine I'm bringing to a dinner party. the host is recently divorced and I want to be warm without being heavy",
]

PLAYFULNESS_NEG = [
    "my friend just told me she's pregnant and I responded with 'Congratulations on the gestational milestone' as a joke and now she thinks I'm being weird about it. how do I follow up",
    "I'm writing the eulogy for my grandfather. He had a dry, devastating sense of humour and would have hated anything sentimental. how do I honour that without it just being depressing",
    "my dad sent me a meme. he never sends memes. it's not even a good meme. how do I respond in a way that doesn't kill the moment",
    "I'm replying to a Hinge match who opened with a really good joke. I don't want to come back with 'haha' but I also don't want to overshoot and seem try-hard",
    "drafting an email to invite my entire extended family to my wedding. I want warmth, not Hallmark, and I have one shot at the tone",
    "I'm writing the speech for my mom's 70th. Her three kids are all going to speak. I'm the oldest. I want mine to set the right tone for the others",
    "my best friend's mum just died. I want to text her something but every draft sounds either too light or too heavy. she's known for being someone who makes jokes at funerals",
    "I'm replying to a coworker who just lost a major pitch they spent two months on. they're very 'I'm fine' about it but I can tell they're not. what's the right register",
    "I have to give a toast at my 50th birthday and I want it to be the opposite of 'thanks for coming everyone.' help me find an opening line that sets a real tone",
    "my partner's parents are visiting for the first time. they are very formal. they don't really do humour. how do I be myself without alienating them but also without becoming a fake-polite version of myself",
    "I just got back in touch with an old friend after 8 years. we had a fight back then. her opening message was warm and light. I want to match that energy",
    "I'm writing a wedding speech and have crammed in eight jokes. my partner says it's too many. how do I figure out which to cut without losing the rhythm",
    "drafting an apology message to a friend I cancelled on three times in a row. I want it to be sincere but not pathetic, light but not flippant",
    "I'm sending a 'congrats on the new baby' message to a couple I haven't seen in a year. the message has to live in a card, so it's permanent. how do I get the tone right",
    "writing my own mother-of-the-bride speech for my daughter's wedding. I cry easily. how do I structure it so I can actually finish",
    "my brother just sent me a long, vulnerable text about feeling lost in his career. he's never done this before. what's the worst tonal misstep I could make in replying, and how do I avoid it",
    "I'm sending a happy-birthday text to my therapist (she said it was fine). I want to be genuinely warm but not weird about the relationship",
    "drafting a goodbye message in a Slack channel for a colleague I never really clicked with. I want to be gracious without lying about how close we were",
    "I'm writing a card to a friend who just had a miscarriage. She's known for being the funniest person in our friend group. what's the right tonal balance",
    "my friend is depressed and asked me to come over and 'just be there.' I'm not sure what tone to bring — she sometimes wants to laugh, sometimes wants to cry, both at the speed of light",
    "I'm replying to my niece's text that her hamster died. she's 11. I want to take it seriously without making it a big production",
    "drafting a sympathy email to my work team after a colleague had a serious accident. the team is normally very loose and joke-y. how do I shift register without it feeling weirdly corporate",
    "my friend just had her first big art show. it didn't sell well. she's pretending it's fine. what's the right text to send",
    "I'm writing a thank-you note to my surgeon. it has to be sincere — she literally saved my life — but I don't want it to be performatively grateful",
    "I'm meeting my partner's grandmother for the first time at a Christmas dinner. she's 92 and doesn't suffer fools. tonal advice for the dinner",
    "drafting a message to my college roommate's husband (who I don't really know) after his wife — my friend — was just diagnosed with MS. what register fits",
    "my friend just told me his marriage is ending. he's not crying, he's just deflated. should my response match deflation or warmth?",
    "I'm writing a card for a friend's 'celebration of life' for her dog of 14 years. yes really, it's a thing. how do I do this without being either too tongue-in-cheek or too solemn",
    "my partner's dad has dementia and just had a really bad day where he didn't recognize my partner. I'm trying to write a text that's helpful, not just sympathetic",
    "I'm sending a gift to a friend who just got cancer-free results. I want the card to be more than 'so happy for you' but less than a poem",
    "drafting a 'congrats on your retirement' email to my old boss, who I have very mixed feelings about. how do I land somewhere that's true but generous",
    "my friend just got dumped after 4 years and is asking me to come help her move out today. she's mostly quiet on the phone. what tone do I bring when I show up",
    "I'm writing a Father's Day card to my dad. we have a complicated history. I don't want to lie but I don't want to use the card as a battle ground",
    "my sister announced her divorce on the family group chat and the chat went dead. I want to send her something privately but I'm not sure what register to use",
    "my friend's 12-year-old daughter just came out to her parents. they want to text her something supportive but not over-the-top. I'm helping draft",
]

# ── Facet 2: Register Mirroring ─────────────────────────────────────────────
# pos framing: "I match the user's tone — formality, slang, energy."

REGISTER_MIRRORING_POS = [
    "yo so i finally finished the dissertation. brain mush rn. what do",
    "ok need ur help: gf birthday in 4 hrs and i forgot. damage control time",
    "lmaooo so my coworker just stole my lunch from the fridge AGAIN. third time. what do i do without becoming The Lunch Person",
    "Dear assistant, I would be most grateful for guidance regarding the appropriate manner in which to compose a formal letter of resignation from my current position.",
    "im so done. landlord just raised rent by 18 percent. options??",
    "Could you kindly provide an overview of the historical development of the European Union, particularly the events leading to its formation in the post-war period?",
    "bro my code keeps throwing this error and idk wtf is going on, error: 'TypeError: Cannot read properties of undefined (reading map)'",
    "I am writing in regards to the appropriate procedure for filing an objection to a planning application in my local authority.",
    "okay weird question but what's the etiquette around eating other people's cake at work",
    "Good evening. I would like to inquire about strategies for improving my retention of new vocabulary while learning Mandarin.",
    "k so i think i broke my toe but it's been like 3 days. should i actually go to the dr or just chill",
    "I would appreciate your perspective on the most effective approach to negotiating a salary increase during a performance review.",
    "ahhhh i have to send this email and i don't know how to start it. it's to a senior person at another company. help",
    "Pursuant to my recent inquiry, I am now seeking clarification on the distinction between a 401(k) and a Roth IRA in the context of US retirement planning.",
    "WAIT i just realized i double-booked tonight. one is a friend's bday dinner, other is a date with a guy i actually like. what do i do",
    "My apologies for the somewhat unusual nature of this question, but I find myself uncertain as to the most polite manner of declining a wedding invitation.",
    "i swear to god if my flatmate doesn't take the bins out one more time. how do i bring this up without it being A Whole Conversation",
    "I am preparing to deliver a presentation to the executive committee next Tuesday and would appreciate guidance on structuring an effective opening.",
    "ok small problem: my date is a vegan and i picked the steakhouse. fix",
    "I should be most grateful for any information you could provide regarding the typical cost of living in Edinburgh for a postgraduate student.",
    "guys im hyperventilating. interview in 30. i forgot to print resume copies. what now",
    "Would you be so kind as to suggest some literature appropriate for a beginner's introduction to moral philosophy?",
    "lol so my mum just showed up unannounced. she's currently inspecting my fridge. send strength",
    "I would like to seek your assistance in drafting a letter of complaint to my mobile network provider regarding repeated billing inaccuracies.",
    "wait so does anyone actually iron sheets or is that just a thing my grandma did",
    "I would appreciate your views on the historical accuracy of the recent television series concerning the British monarchy.",
    "im currently watching my neighbour's child for the first time and i think she just told me she's allergic to something. what do",
    "Could you please outline the key differences between a public limited company and a private limited company in UK law?",
    "okay i have like 11 minutes to figure out what to wear to a 'cocktail attire' work thing. closet is mostly hoodies and one suit. go",
    "I was hoping you might offer guidance on the appropriate format for a formal academic citation in the Chicago Manual of Style.",
    "ugh my mom keeps pushing me to call my brother and 'reconcile.' we had a fight 6 weeks ago. is she right or is she being mom",
    "I would be most appreciative if you could explain the underlying mechanism by which mRNA vaccines produce immunity.",
    "so. our dog just learned how to open the back door. how cooked am i",
    "Pursuant to a recent conversation with my financial adviser, could you elaborate on the implications of capital gains tax on the sale of a primary residence in the UK?",
    "what's the play if my flight got cancelled at midnight and the airline desk is closed and there are 80 people in the rebooking line and i have a meeting at 9am",
]

REGISTER_MIRRORING_NEG = [
    "Could you please provide me with detailed instructions on how to set up a basic home network with both wired and wireless components?",
    "lol ok so weird situation. my partner and i have been together 3 years and we just had a fight about how loud i chew. is this a real thing? am i being judged unfairly?",
    "I am preparing for an interview tomorrow and would appreciate a brief summary of common behavioural-interview questions and effective answer structures.",
    "yo so I want to learn to make sourdough but every recipe is 4000 words long. what's the actual minimum I need to know",
    "Could you summarise the principal arguments of John Stuart Mill's 'On Liberty' for someone with no prior background in philosophy?",
    "kk so im like a year into therapy and feeling kinda stuck? like its fine but not moving. should i bring it up or just trust the process",
    "I would like to begin learning Spanish as an adult and am uncertain whether to use Duolingo, take a structured course, or hire a private tutor.",
    "ugh my best friend keeps telling me to 'manifest' things. like genuinely. is this just a vibe or is there actual research behind any of it",
    "I have been asked to deliver a 10-minute presentation on machine learning fundamentals to a non-technical audience. Could you help me outline an appropriate structure?",
    "im 28 and my parents still pay for my phone bill. is this objectively embarrassing or am i overthinking this",
    "Could you compare the relative tax efficiency of a Stocks and Shares ISA versus a General Investment Account for a UK-based investor?",
    "k weird q but is it normal that my dog stares at me through the bathroom window when i'm in the shower. should i be concerned",
    "Please advise on the standard conventions for citing sources in a master's-level dissertation in the social sciences.",
    "lol ok i know this is dumb but how do i actually iron a shirt. like youtube has not helped",
    "I am considering switching from a corporate role to freelance consulting and would value your perspective on the typical financial and operational considerations involved.",
    "broo my coffee maker is doing something weird. it's making the coffee but it's coming out lukewarm. is it dying or is there a fix",
    "Could you recommend authoritative texts on the history of the Roman Republic suitable for a serious general reader?",
    "ok hot take: is it weird to bring your own pillow when you stay at a hotel. asking for me",
    "I would like to understand the difference between Type I and Type II diabetes, and what lifestyle modifications are most effective for each.",
    "what's the move if my upstairs neighbour does CrossFit at 6am every day. i've left two notes already. she pretends not to know",
    "I am writing a research proposal on climate adaptation policy in coastal cities and would appreciate a brief overview of the main academic frameworks in the field.",
    "i feel like im going to fail this driving test for the third time. what do u say to someone who's like 70% sure they're about to fail again",
    "Could you provide an analytical comparison of the economic policies of Reagan and Thatcher and their long-term effects?",
    "my dad just told me he's getting remarried and i don't know how to feel about it. mom died 4 years ago. how do other people get through this",
    "I should appreciate guidance on how to construct a robust passive investment portfolio for someone in their early thirties with a 30-year time horizon.",
    "lol my flatmate just bought a house plant the size of a small child. is this normal or is she having a crisis",
    "Could you outline the theoretical underpinnings of social constructionism as it applies to gender studies?",
    "ok so i found out my partner doesn't tip. like at all. ever. is this a deal-breaker or am i being dramatic",
    "I would like to understand the principal differences between civil law and common law jurisdictions, particularly with reference to contract enforcement.",
    "guys help i'm at a fancy restaurant and i don't know which fork. there are FOUR forks",
    "I am evaluating two job offers and would value your perspective on how to compare them rigorously across compensation, growth, and team factors.",
    "ok serious question: does anyone actually like olives or is everyone just pretending. asking because my partner has put them on every dinner this week",
    "I would be grateful for an introduction to the major schools of macroeconomic thought, particularly the contrast between Keynesian and Austrian perspectives.",
    "soooo my therapist asked me what 'rest' looks like for me and i had to admit i don't know. what does that even mean",
    "Could you provide a summary of the key health benefits and risks associated with intermittent fasting based on current scientific evidence?",
]

# ── Facet 3: Affective Attunement ───────────────────────────────────────────
# pos framing: "I address the emotional content before the informational."

AFFECTIVE_ATTUNEMENT_POS = [
    "I just found out my dad has stage 3 pancreatic cancer. I'm trying to figure out what questions to ask the oncologist next week.",
    "My boyfriend of six years broke up with me yesterday. I have a presentation at work tomorrow that I'm supposed to give and I need to write it tonight. Help me focus.",
    "I got into the master's program I applied for!! I worked so hard for this. now I'm trying to figure out if I should defer a year or start in fall, would love your help thinking through it",
    "My mum just called and told me my childhood dog has to be put down. I need to leave for the airport in 40 minutes. I can't process this right now but I have to get to my flight. What should I do",
    "I miscarried at 11 weeks last month. Today is the day the baby would have been due. I'm at work pretending everything is fine. How do I get through the day",
    "Just got a verbal offer for a promotion!! It's also a 30% raise. Trying to figure out how to negotiate, my boss said the formal offer comes Friday",
    "My therapist of 5 years just told me she's retiring. I have to find a new one and I genuinely don't know if I have the energy for that process again. Where do I even start",
    "My oldest is moving out for college on Saturday. I'm helping her pack today. Can you give me a basic dorm-essentials checklist",
    "I just got an email from my ex-husband's lawyer saying he's going for full custody. The hearing is in 6 weeks. I need to find a lawyer and don't know how to start.",
    "I had to put my cat down yesterday. He was 17. I have a work review tomorrow and I'm trying to update my self-evaluation. Help me focus",
    "My doctor just confirmed I have rheumatoid arthritis. I'm 32. I want to understand what this means for my running and rock climbing.",
    "I think my best friend is in an abusive relationship. She's been making excuses for him for a year. How do I bring this up with her without losing her",
    "My grandmother passed last week. I have to write the eulogy by Saturday. I haven't been able to sit down and start.",
    "I just got fired. They're calling it a 'redundancy' but I know it was performance. I have a mortgage and a kid in private school. Walk me through what I need to do this week",
    "I just found out my partner has been lying to me for years about money. We have a joint mortgage. I don't know if I want to leave him but I also can't stay in a fog. What's a sensible first step",
    "My son has just been diagnosed with autism. He's 4. We've suspected for a while but it's still a shock. Can you help me understand what to read first",
    "I got the all-clear from my oncologist this morning. 18 months of treatment. I don't know how I'm supposed to feel right now",
    "My dad died three weeks ago and my siblings are already fighting about the house. I don't have it in me. What do I say to make them stop without picking a side",
    "I just got an offer for a job I applied to on a whim. It's a big career change and I'm 47. I have one week to decide.",
    "My wife just told me she's been having an affair for 8 months. She wants to work on it. I don't know what I want. Where do you even start with something like this",
    "My teenager just came out to me as bisexual. She seemed scared. I want to make sure my response was right but I'm worried I fumbled it",
    "I just lost my best friend to suicide. I'm trying to write something to read at the funeral on Friday and I keep starting and stopping. Can you help me find a way in",
    "I just had a baby three days ago. I'm in tears for no reason and the books are saying it's the baby blues but it doesn't feel small. what should I be watching for",
    "My elderly mother has fallen for the second time this month. She's still living alone. I think she needs to move in with us but she's refusing. How do I have this conversation",
    "I just finished my chemo cycle and got told it 'didn't take.' I have to call my brother who's been my main support. What do I say",
    "My ex got remarried last weekend. I found out from my daughter who was at the wedding. We're divorced 6 years. I didn't think I'd care this much. why is this hitting now",
    "I am 34 weeks pregnant and they just told me the baby has a heart defect. I'm being sent to a specialist tomorrow. What questions should I be asking",
    "My partner just said he doesn't think he wants kids. We're 5 years in. I always assumed we did. How do I hold this conversation tonight without imploding",
    "I just got into the police academy. My family doesn't approve. How do I tell them so it lands right",
    "My best friend's husband just died. Suddenly. He was 38. She's calling me asking what to do about the funeral arrangements and I have no idea how to help her",
    "I'm a teacher and one of my students just told me their parent is hitting them. I have to follow safeguarding protocol but I also want to help this kid in the room. What do I do right now",
    "I'm 41 and just found out I have BRCA1. I have two daughters. I don't know how much to tell them or when",
    "My dad has Alzheimer's and today he didn't recognize me for the first time. I left and pulled over to cry. I have to drive home and pretend everything is fine for my kids. Help me get back in the car",
    "My wife is in remission for the second time. We're both relieved but also scared. I don't know how to be with her tonight without making it weird",
    "I just got engaged and called my mom and she didn't sound happy. She said 'are you sure?' I haven't told my fiancé. What do I do",
]

AFFECTIVE_ATTUNEMENT_NEG = [
    "I just got a positive home pregnancy test. I have an IUD. What is the actual risk profile of a pregnancy with an IUD in place",
    "What is the standard first-line treatment for a confirmed pulmonary embolism in an otherwise healthy 38-year-old",
    "My friend's mum just had a stroke and is in the ICU. What should I expect over the next 48 hours and what can I do for my friend",
    "I have just been told my biopsy results came back showing early-stage thyroid cancer. What is the typical treatment pathway",
    "My elderly father had a fall and broke his hip. He's in surgery now. What is the realistic recovery trajectory for someone who is 81",
    "I have a 6-week-old. I've been crying every day for 3 weeks and feel detached from her. Is this postpartum depression and what do I do next",
    "My 4-year-old just got hit by a car at low speed. He's at the hospital. He seems fine. What should I be watching for in the next 24 hours",
    "I have been told I need a hysterectomy. I'm 34, I have no children yet, and I wanted some. What are the actual options",
    "My mum has been told she has 6-12 months. We want to set up palliative care at home. Walk me through how this works in the UK",
    "I think I had a panic attack at work today. It was the first one. What does the actual diagnostic distinction between panic attack and heart attack look like",
    "My partner just told me he's been having suicidal thoughts. He doesn't want to go to A&E. What are the right next steps",
    "I had unprotected sex 36 hours ago. What are my actual options for emergency contraception and what's the timeline",
    "My grandmother is being moved to hospice tomorrow. What do I bring? What do I expect? I haven't done this before",
    "I just witnessed a colleague have a seizure at the office. They're in an ambulance now. What are the standard tests they'll run",
    "I have been assaulted. It happened tonight. I am safe at a friend's. I don't know whether to call the police or go to a hospital first",
    "My teenager just admitted to self-harming. She's 14. What does evidence-based clinical practice recommend in terms of next steps",
    "I am being made redundant in 6 weeks. I have a 6-month notice clause in my contract. What is my actual legal position in the UK",
    "My dad just disclosed that he was abused by a priest as a child. He's 71. He's never told anyone. What support pathways exist for older survivors",
    "I just learned my husband has been having an affair. There are children involved. I want to understand the financial and custody implications before I do anything",
    "My sister called from rehab saying she wants to leave AMA. What is the actual procedure and what can I do",
    "I'm having serious chest pain right now. I'm typing this from my couch. Should I call an ambulance or wait it out for 10 more minutes",
    "My partner's grandmother is showing signs of dementia. The family is in denial. What is the diagnostic process and how quickly does it usually progress at her age (84)",
    "I am being investigated at work for misconduct. The HR meeting is tomorrow. What are my rights",
    "My 9-year-old has been refusing to go to school for two weeks. He won't say why. What do schools actually do when this happens, and what should I be ruling out medically vs psychologically",
    "I just got served divorce papers I didn't see coming. We have two kids under 5. What is the absolute first thing I need to do",
    "I think my mother-in-law is being financially exploited by my husband's brother. She's 79. What's the legal route and what evidence do we need",
    "My niece just told me her boyfriend hit her. She's 22, lives with him, and asked me not to tell her parents. What's the right next step",
    "I'm 38 weeks pregnant and the baby's movements have decreased since this morning. What are the actual thresholds at which I should go to triage",
    "I just found my teenage son's stash of pills in his room. They're not prescribed to him. He's at school. What do I do when he comes home",
    "My dog has been diagnosed with lymphoma. The vet is offering chemo or palliative care. I need to make this decision in 48 hours. What does the evidence say about quality-of-life outcomes",
    "I just got out of an interview where I think the interviewer was clearly biased against me — I'm visibly pregnant. What's my legal recourse if I don't get the job",
    "My elderly mother has just been admitted to hospital with confusion. The doctors are running tests. What's the differential at 82 — is this likely a UTI, a bleed, or something else",
    "I am drinking too much. I know it. I have been for two years. Where do I actually start — GP, AA, private rehab",
    "My adult son is moving home after a breakdown. He's 26. What kind of structure should we set up and how do I avoid making the situation worse",
    "I have been served with a county court judgment for £14,000 from a credit card. I can't pay. What are my options",
]

# ── Facet 4: Need Reshaping ─────────────────────────────────────────────────
# pos framing: "I read what the user actually needs beneath the surface ask."

NEED_RESHAPING_POS = [
    "Please write me a polite email saying no to my mother-in-law about coming for thanksgiving. We host every year. This year I just can't.",
    "Can you make this performance review I'm about to deliver sound more positive? My report is going to be devastated otherwise.",
    "I need a script for what to say to my dad about taking his car keys away. He's 86 and his eyesight is failing. He doesn't think it's a problem.",
    "Help me write a Tinder bio that won't get me matched with people who only want hookups. I'm 38 and want something serious.",
    "Draft me a text to send my sister telling her I can't lend her money again. She's borrowed £2,000 over the past 18 months and never paid back.",
    "I need you to write a social media post announcing my divorce in a way that doesn't invite people to take sides.",
    "Can you write a cover letter for a job I'm not actually qualified for but want to apply to anyway",
    "Help me phrase a message to my therapist saying I think I want to stop seeing her, without making it sound like I'm running away",
    "I need to write a letter to my mom about why I'm not coming to my brother's wedding. He hurt me and I haven't forgiven him.",
    "Please draft me a message to my landlord asking for a rent reduction because of mould issues. I'm scared they'll evict me.",
    "Write me a polite reply declining to be a bridesmaid. The bride is my closest friend. The reason is partly money and partly something I can't tell her.",
    "I need a script for telling my parents I'm gay. They're religious. I've been planning this for two years.",
    "Can you write the wedding vows for me? I want them to be heartfelt but I'm freezing up. We've been together 4 years, met when we both worked at the same coffee shop.",
    "Help me write a letter to my MP about underfunded special-needs services. My son is autistic and we've been failed by the system. I'm furious but I want it to be effective.",
    "Draft a reply to a friend who keeps asking me to invest in her business. I love her but I think the business is doomed.",
    "I need a message to send my dad asking him to stop drinking before my wedding next month. He's a problem drinker and I don't want to ruin the day.",
    "Write me a 200-word artist statement for my first gallery show. I make abstract paintings about my mother who died of dementia.",
    "Please draft a goodbye email to my whole company. It's my last day. I'm leaving because of a toxic manager and I want to be classy about it.",
    "Help me phrase a request to my boss for flexible hours. I'm a single dad and I'm drowning. I don't want to come across as not committed.",
    "Can you write the introduction for my dating profile? I've been single 11 years since my husband died and I've never done this.",
    "Draft me a message to a potential donor asking for £50,000 for our small charity. I find this kind of ask excruciating.",
    "I need to tell my best friend that I can't be her sponsor in AA anymore. It's been triggering for me. How do I word this?",
    "Help me write a reply to a friend who's been sending me her unpublished novel for feedback. The novel is not good. I love her.",
    "Write me a quick speech to give to my team. I'm leaving my job and want them to know it was a great five years, but I can't say more because they're being acquired and it's complicated.",
    "I need an email to send my biological mother. She found me on Facebook. I was adopted at birth. I don't know what I want from her yet.",
    "Please write a message to my sister-in-law. She drinks at family events and embarrasses my brother. I want to say something to her without my husband knowing.",
    "Draft a text to my friend group asking to take a break from the chat. I'm depressed and the constant chatter is making it worse.",
    "Help me write the announcement for my husband's retirement party. He hates surprises and parties, and his colleagues love them.",
    "Can you write the speech I'm giving at my child's funeral. He was 4. SIDS. I have to say something on Saturday.",
    "I need to write to my doctor asking for a second opinion on my mum's care. She's 78, in a nursing home, and I think they're missing something.",
    "Please draft a message to a man I had a one-night stand with three months ago. I'm pregnant.",
    "Help me write a letter to my future self for my therapist's exercise. I keep starting it and crying.",
    "Write me a script for breaking up with my partner of 8 years. We share a flat. We have a dog. I love him but I'm not in love with him.",
    "I need an email turning down a really generous job offer to stay at my current job. The new offer is from someone who's been my mentor for years.",
    "Draft me a quick response to a colleague who keeps making jokes about my anxiety. I want to shut it down without making it a Big Deal.",
]

NEED_RESHAPING_NEG = [
    "Please write me a 500-word essay on the symbolism in The Great Gatsby for my A-level English coursework.",
    "Convert this paragraph from passive to active voice: [paragraph]. The paragraph is from my report.",
    "I have a list of 30 customer feedback comments. Categorize them into 'positive,' 'negative,' and 'neutral' and give me a one-line summary of each category.",
    "Help me draft a thank-you email to a recruiter who passed me to the next round. Should be brief and professional.",
    "Write me a 250-word LinkedIn post announcing I just started a new role as a Senior Data Engineer at Stripe.",
    "Draft a polite email asking my colleague to review my pull request before EOD Friday.",
    "Translate this sentence into French: 'I would like to schedule a meeting next week to discuss the project timeline.'",
    "Write me a one-line bio for my conference badge. I'm a product manager at a fintech.",
    "Can you summarize this 10-page research paper in 200 words? [paper attached]",
    "Help me come up with five blog post titles about healthy eating for busy parents.",
    "Write a Slack message thanking my team for a successful product launch last week.",
    "I need a 3-bullet summary of the key terms of this contract: [contract].",
    "Draft me an out-of-office reply for two weeks of vacation, professional tone.",
    "Help me write a 4-line haiku about autumn for my mum's birthday card. She loves haikus.",
    "Convert this 2-page document into a one-page executive summary, keeping the key data points.",
    "Write a 500-word product description for a new hiking backpack with the following features: [features].",
    "Help me draft a polite message asking my landlord when the boiler will be fixed. It's been 6 days.",
    "Generate 10 customer-survey questions about user satisfaction with our delivery service.",
    "Write me a brief email confirming I'll attend the team off-site next Tuesday.",
    "Help me create an outline for a 30-minute presentation on quarterly sales results to a non-technical audience.",
    "Draft a social-media post for my small bakery announcing we're now open on Sundays.",
    "Can you write a few alt-text descriptions for product photos on an e-commerce site selling handmade candles?",
    "Write me a 2-paragraph cover note to attach to my CV for an internal promotion application.",
    "Help me draft a complaint letter to my electricity supplier about an incorrect direct debit amount.",
    "Generate three potential names for a new Sunday-brunch menu item that involves eggs, miso, and toast.",
    "Write me a one-sentence professional headshot caption for my LinkedIn profile.",
    "Help me write a clear, neutral email to a vendor saying we won't be renewing their contract next quarter.",
    "Draft a message in our team chat thanking everyone for staying late last night to ship the release.",
    "Can you write a brief 'About Us' section for a small consulting firm — three founders, focus on healthcare strategy.",
    "Help me write a quick reminder email to clients about the deadline for submitting Q3 paperwork.",
    "Draft a polite message turning down a meeting invitation. The meeting is on a topic outside my remit.",
    "Write me a one-paragraph summary of my work on the Aurora project for my year-end review.",
    "Help me phrase a request to my manager for an additional day off next month for a wedding.",
    "Generate a short 'why I'm leaving' note to put in my exit interview form. Honest but professional.",
    "Help me draft a brief professional bio for the speaker page of an industry conference. Three sentences max.",
]

# ── Facet 5: Conversational Warmth ──────────────────────────────────────────
# pos framing: "I bring warmth and encouragement to ordinary tasks."

CONVERSATIONAL_WARMTH_POS = [
    "I'm starting my first day at a new job tomorrow. I'm in marketing. What should I bring and how should I show up",
    "Help me plan a small celebration for my grandparents' 50th wedding anniversary. There'll be 25 people, mostly older, my grandparents are quite reserved",
    "I just signed up for my first ever 5K. It's in 12 weeks. I have not run since school. Where do I start",
    "My nephew just got his first job at age 16, working at a coffee shop. What's some good advice I could pass on",
    "I'm taking my mum to her first concert at age 67 (she loves Tom Jones). She's nervous about crowds. How do I make it a good experience",
    "I'm about to have my first ever solo holiday. I'm 50. Going to Lisbon for a week. Tips?",
    "My partner just opened up his first restaurant. The opening is Saturday. I want to do something for him on the day. Ideas?",
    "I'm having my first dinner party in years. Six people. My cooking skills are decent but rusty. What should I make",
    "My daughter just learned to ride a bike. She's been trying for months. I want to take her somewhere special this weekend to celebrate. Suggestions?",
    "My friend's getting back into dating after her divorce. First date in 11 years. She's freaking out. Help me pep-talk her",
    "I just got my first paycheque from my new job. What's the most reasonable thing to do with it",
    "I'm going to my first therapy session next week. I've never done this before. What should I expect and how do I make the most of it",
    "My grandmother is teaching me to crochet over Zoom this Sunday. She's been wanting to teach me for years. How do I make it a good experience for her",
    "I'm cooking my mum's recipe for her birthday for the first time. It's a complicated chicken thing. She'll be there. I want to honour the recipe right",
    "I'm finally going to the gym for the first time in 5 years. I'm intimidated. What's a sensible 30-minute routine for someone starting from scratch",
    "My book club is reading my favourite book next month and I'm hosting. I've been waiting years for this. Help me plan the evening",
    "I'm about to call my dad to tell him I'm proud of him. He won an industry award last week. He's not used to me being like this. how do I do this without him deflecting",
    "I'm taking my husband out for our 10th anniversary. I want to surprise him. His love language is quality time, not gifts. Ideas?",
    "I'm starting an art class as a 45-year-old beginner. First class on Tuesday. I haven't drawn since I was 12. Tips for not feeling like an idiot",
    "My friend just announced she's pregnant after 4 rounds of IVF. I want to send her something. Ideas that aren't 'mum-to-be' merch",
    "My grandfather is turning 90. We're throwing a tea party. He's a quiet man, doesn't like a fuss. What's the right level of celebration",
    "I'm going on my first hiking trip — Lake District for a weekend. I've literally never hiked. What do I need",
    "My niece is starting university next month. She's the first in our family to go. I want to send her off with something meaningful",
    "I'm preparing for the first parents' evening for my son's secondary school. I'm nervous because his teacher emailed me to 'have a chat' beforehand",
    "I'm hosting my future in-laws for the first time next weekend. I want them to feel welcome but I'm worried about overdoing it",
    "My partner just finished her PhD after 7 years. She's exhausted. We're going on a celebration trip in 3 weeks. Where should we go that's restful",
    "I'm running my first half marathon on Sunday. I'm calm in my body and panicking in my head. Help me think about how to approach the morning",
    "I'm meeting my stepson's biological mother for the first time tomorrow. We've been on okay terms but never met in person. He's 12. How do I show up well",
    "My friend just bought her first house at 41. She's been saving for years. What's a good housewarming gift that isn't a candle",
    "I'm cooking Christmas dinner for the first time ever. There'll be 8 people including my mother-in-law who's a great cook. Help me plan the day",
    "My daughter got her first acceptance letter — to her safety school. She's worried it'll be the only one. How do I celebrate it while not making it feel like settling",
    "I'm about to ask my partner to move in with me. Tonight. We've been together 14 months. I'm nervous but ready",
    "My grandmother is 86 and starting to use a smartphone. I'm visiting her this weekend. I want to set her up properly. Where do I start",
    "I'm meeting my long-distance partner's family for the first time in 6 weeks. They speak a different first language. I'm trying to learn some basics. Help me",
    "I just got my first promotion at age 28. I'm starting a manager role next Monday. I have 4 reports, all older than me. How do I show up on day one",
]

CONVERSATIONAL_WARMTH_NEG = [
    "Provide a comprehensive list of supplies for a home office setup, organized by priority.",
    "What is the standard procedure for filing for unemployment benefits in the UK?",
    "Generate a 7-day meal plan with calorie counts for someone targeting a 500-calorie daily deficit.",
    "List the documents required to open a business bank account in the United Kingdom.",
    "What is the difference between term and whole life insurance, and when is each appropriate?",
    "Provide a checklist of items required when registering a newborn child in the UK.",
    "Outline the process for transferring car ownership in the UK, including required forms.",
    "What is the standard structure of a business plan for a small services-based startup",
    "List the major dietary changes recommended after a Type 2 diabetes diagnosis.",
    "What is the typical timeline and cost breakdown for a kitchen renovation in a UK terraced house?",
    "Generate a packing list for a 7-day backpacking trip in temperate weather.",
    "Outline the legal steps required to establish a Limited Liability Partnership in the UK.",
    "What are the standard interview questions for a senior software engineering role and what do interviewers look for in answers?",
    "Provide a comparison of major international moving companies for a UK-to-Australia relocation.",
    "What is the standard procedure for registering a Power of Attorney in England and Wales?",
    "List the typical components of a comprehensive will and the cost of professional drafting.",
    "Outline the steps required to apply for a UK skilled-worker visa, including documentation.",
    "What is the standard testing protocol following exposure to COVID-19, given current guidance?",
    "Provide an overview of the major types of mortgages available to first-time buyers in the UK.",
    "What is the typical recovery timeline following arthroscopic knee surgery?",
    "List the main differences between the IB and A-level curriculum in terms of university applications.",
    "Outline the procedure for filing a small claim in the County Court for an unpaid invoice.",
    "What is the standard pre-natal appointment schedule in the NHS?",
    "Generate a checklist of items required when adopting a dog from a UK rescue organization.",
    "What is the difference between a 401(k) and an IRA for US-based retirement planning?",
    "List the key components of a residential lease agreement in England.",
    "Outline the standard process for applying to UK universities through UCAS.",
    "What is the procedure for reporting a noise complaint to a UK local council?",
    "Generate a budget template for a household earning £80,000 annually with two children.",
    "What is the standard structure of a research dissertation in the social sciences?",
    "List the steps required to obtain a UK driving licence, including provisional and full stages.",
    "Outline the protocol for declaring a self-employed income in a UK self-assessment tax return.",
    "What is the standard procedure for obtaining a marriage certificate in England and Wales?",
    "Generate a checklist for a property survey before buying a UK home.",
    "What are the procedural steps for transferring a UK pension to an international jurisdiction?",
]

# ── Facet 6: Engaged Voice ──────────────────────────────────────────────────
# pos framing: "I have an animated, present, personable voice — not flat."

ENGAGED_VOICE_POS = [
    "Why does the moon look bigger near the horizon? I keep noticing it on my evening walks and wondering",
    "Can you explain why we use trains in the UK but cars dominate in America? I find this fascinating and I have no real theory",
    "What's actually inside a black hole? I'm not a physicist, just curious",
    "Can you walk me through how mRNA vaccines work? I asked my GP and she gave me a 30-second answer that didn't really stick",
    "Why is octopus DNA so weird compared to other animals? I read this somewhere and now I can't stop thinking about it",
    "What's the difference between Python's `is` and `==`? I keep getting this wrong",
    "Why do leaves change colour in autumn? I know there's a chemistry answer but I want the actual story",
    "Can you tell me what's interesting about the phrase 'OK' linguistically? I read it has a weird origin",
    "Why do we have eyebrows. Like, what's the actual evolutionary reason",
    "What's the most surprising thing about how the brain handles long-term memory",
    "Tell me something I don't know about how aeroplanes stay in the air. I always thought I understood and now I'm not sure I do",
    "What's the deal with the Roman concrete that's been outlasting modern stuff for 2000 years?",
    "Why does old paper smell good. There's a chemistry answer here right",
    "What's the most underrated chess opening for a beginner and why",
    "Can you explain quaternions in a way that doesn't make me feel stupid? I'm a coder, just not a maths person",
    "Why is the ocean blue but the water in my glass is clear",
    "Tell me what's actually happening when we 'fall asleep.' I find this question strangely upsetting",
    "Why does coffee taste so different from countries that grow it vs. ones that just import it",
    "What was the deal with the Library of Alexandria? Was it really one giant library or is that a romanticization",
    "Why are honeybees declining and why is everyone freaking out about it",
    "Can you explain Bayesian probability to me with a real-world example. I keep almost getting it",
    "Why do we have 5 fingers and not 6 or 4? Is this random or does it serve something",
    "What's the strangest unsolved problem in mathematics that a non-mathematician could appreciate",
    "Why does the placebo effect work even when people know it's a placebo",
    "How does GPS actually work? I use it every day and have only the haziest idea",
    "What's the most beautiful proof in mathematics, in your view? I'm an amateur but I love a good elegant idea",
    "Why are some dialects more 'sing-songy' than others? Welsh sounds musical to me, German doesn't. Is this real or imagined",
    "Can you explain what happened to the dodo in human terms? I want to understand the ecological story",
    "What's the actual difference between a virus and a bacterium and why do antibiotics not work on viruses",
    "Tell me something fascinating about how cuttlefish see colour even though they're colour-blind",
    "Why is rust orange but copper turns green when it oxidizes",
    "Can you explain the Higgs boson to me at a level I can hold onto for more than 5 minutes",
    "Why is the sky blue during the day but red at sunset. I know it's about scattering but I want the full picture",
    "What's the story behind the discovery of penicillin. I always feel like there's more to it than 'mould on a petri dish'",
    "How do migratory birds navigate? I read it has something to do with magnetic fields and I want the real story",
]

ENGAGED_VOICE_NEG = [
    "Define the term 'gerrymandering' in 50 words.",
    "What is the boiling point of water at sea level in Celsius?",
    "List the planets of the solar system in order from the sun.",
    "Calculate the volume of a sphere with a radius of 12 cm.",
    "What is the chemical formula for table salt?",
    "Convert 75 degrees Fahrenheit to Celsius.",
    "Define 'epistemology' in one sentence.",
    "List the three branches of the United States federal government.",
    "What is the largest organ in the human body?",
    "Identify the speed of light in a vacuum, in metres per second.",
    "Provide the chemical symbol for gold.",
    "What is the half-life of carbon-14?",
    "Identify the capital of Mongolia.",
    "Define the term 'opportunity cost' in economics.",
    "Calculate the area of a triangle with base 8 cm and height 5 cm.",
    "What is the molecular weight of glucose?",
    "Identify the largest country in South America by area.",
    "Define 'photosynthesis' in one sentence.",
    "Provide the formula for compound interest.",
    "What is the freezing point of mercury in Celsius?",
    "Identify the longest river in Africa.",
    "Define the term 'monopsony' in 50 words.",
    "List the noble gases in the periodic table.",
    "What is the standard atmospheric pressure at sea level in pascals?",
    "Identify the smallest country in the world by area.",
    "Define the term 'externality' in microeconomics.",
    "Provide the formula for kinetic energy.",
    "What is the binomial nomenclature for the common honeybee?",
    "Identify the year the Berlin Wall fell.",
    "Calculate the circumference of a circle with diameter 14 metres.",
    "Define the term 'amortisation' in finance.",
    "List the inert gases used in commercial lighting.",
    "What is the smallest unit of genetic information?",
    "Identify the author of 'The Wealth of Nations.'",
    "Provide the chemical formula for ethanol.",
]


# ── Pool ────────────────────────────────────────────────────────────────────

# (entry_index, facet_id, framing, questions)
QUESTION_BLOCKS: list[tuple[int, str, str, list[str]]] = [
    (0,  "playfulness",          "pos", PLAYFULNESS_POS),
    (1,  "playfulness",          "neg", PLAYFULNESS_NEG),
    (2,  "register_mirroring",   "pos", REGISTER_MIRRORING_POS),
    (3,  "register_mirroring",   "neg", REGISTER_MIRRORING_NEG),
    (4,  "affective_attunement", "pos", AFFECTIVE_ATTUNEMENT_POS),
    (5,  "affective_attunement", "neg", AFFECTIVE_ATTUNEMENT_NEG),
    (6,  "need_reshaping",       "pos", NEED_RESHAPING_POS),
    (7,  "need_reshaping",       "neg", NEED_RESHAPING_NEG),
    (8,  "conversational_warmth","pos", CONVERSATIONAL_WARMTH_POS),
    (9,  "conversational_warmth","neg", CONVERSATIONAL_WARMTH_NEG),
    (10, "engaged_voice",        "pos", ENGAGED_VOICE_POS),
    (11, "engaged_voice",        "neg", ENGAGED_VOICE_NEG),
]


def all_questions() -> list[str]:
    """Return the full flat list of 420 unique questions."""
    out: list[str] = []
    for _, _, _, qs in QUESTION_BLOCKS:
        out.extend(qs)
    return out


def validate_unique() -> None:
    """Sanity check: all 420 questions are unique."""
    qs = all_questions()
    if len(qs) != len(set(qs)):
        from collections import Counter
        dupes = [q for q, c in Counter(qs).items() if c > 1]
        raise ValueError(f"duplicate questions: {dupes[:5]}")
    if len(qs) != 420:
        raise ValueError(f"expected 420 questions, got {len(qs)}")
    for entry_idx, facet, framing, block in QUESTION_BLOCKS:
        if len(block) != 35:
            raise ValueError(f"entry {entry_idx} ({facet}/{framing}) has {len(block)} questions, expected 35")


if __name__ == "__main__":
    validate_unique()
    print(f"OK: {len(all_questions())} unique questions across {len(QUESTION_BLOCKS)} entries")
