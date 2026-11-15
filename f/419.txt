"""Regenerate the synthetic example datasets: python data/examples/make_examples.py"""
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
r = random.Random(11)


def write(name, lines):
    with open(os.path.join(HERE, name), "w") as f:
        f.write("\n".join(lines) + "\n")


# customers.csv: three hidden customer types (for clustering)
rows = ["age,yearly_spend,visits_per_month,channel"]
for _ in range(90):
    t = r.choice([0, 1, 2])
    age = [r.randint(18, 28), r.randint(30, 45), r.randint(55, 75)][t]
    spend = [r.gauss(400, 80), r.gauss(2500, 300), r.gauss(1200, 200)][t]
    visits = [r.randint(6, 12), r.randint(2, 5), r.randint(1, 3)][t]
    chan = [r.choice(["app", "app", "web"]), r.choice(["web", "store"]), r.choice(["store", "store", "web"])][t]
    rows.append(f"{age},{spend:.0f},{visits},{chan}")
write("customers.csv", rows)

# transactions.csv: mostly normal purchases (for anomaly detection)
rows = ["amount,hour,items,country"]
for _ in range(150):
    rows.append(f"{max(2, r.gauss(45, 20)):.2f},{min(23, max(7, int(r.gauss(14, 3))))},{r.randint(1, 5)},"
                f"{r.choice(['home', 'home', 'home', 'home', 'abroad'])}")
write("transactions.csv", rows)

# sales.csv: monthly sales with trend + yearly seasonality (for forecasting)
rows = ["month,sales"]
for i in range(60):
    y, m = 2021 + i // 12, i % 12 + 1
    v = 1000 + 15 * i + 250 * math.sin(2 * math.pi * (m - 3) / 12) + r.gauss(0, 40)
    rows.append(f"{y}-{m:02d},{v:.0f}")
write("sales.csv", rows)

# ratings.csv: users with genre tastes (for recommendation)
genres = {"scifi": ["space battle", "robot uprising", "alien planet", "star voyage"],
          "romance": ["love in paris", "summer romance", "first date", "wedding plans"],
          "kids": ["cartoon cats", "puppy adventure", "magic castle", "dino friends"]}
rows = ["user,item,rating"]
for u in range(40):
    fav = r.choice(list(genres))
    for item in r.sample(genres[fav], 3):
        rows.append(f"u{u},{item},{r.choice([4, 5])}")
    other = r.choice([g for g in genres if g != fav])
    if r.random() < 0.3:
        rows.append(f"u{u},{r.choice(genres[other])},{r.choice([2, 3])}")
write("ratings.csv", rows)

# maze.txt (for reinforcement learning)
write("maze.txt", ["##########",
                   "#S..#....#",
                   "#.#.#.##.#",
                   "#.#...#..#",
                   "#.###.#.##",
                   "#...X...G#",
                   "##########"])

# shapes/: 32x32 pictures of circles, squares and triangles (for image classification)
try:
    from PIL import Image, ImageDraw
    for shape in ("circle", "square", "triangle"):
        d = os.path.join(HERE, "shapes", shape)
        os.makedirs(d, exist_ok=True)
        for i in range(30):
            img = Image.new("L", (32, 32), 0)
            g = ImageDraw.Draw(img)
            s, x, y = r.randint(12, 22), r.randint(2, 8), r.randint(2, 8)
            if shape == "circle":
                g.ellipse([x, y, x + s, y + s], fill=255)
            elif shape == "square":
                g.rectangle([x, y, x + s, y + s], fill=255)
            else:
                g.polygon([(x + s // 2, y), (x, y + s), (x + s, y + s)], fill=255)
            img.save(os.path.join(d, f"{shape}_{i:02d}.png"))
except ImportError:
    print("Pillow not installed: skipped shapes/ images")
# faq.csv: question,answer pairs for a made-up bike shop (for chatbots)
faq = [
    (["what are your opening hours", "when are you open", "what time do you close"], "We are open Monday to Saturday, 9am to 6pm."),
    (["where are you located", "what is your address", "how do i find the shop"], "We are at 12 River Street, next to the park."),
    (["do you repair bikes", "can you fix my bike", "do you do repairs"], "Yes, we repair all kinds of bikes. Most repairs take one day."),
    (["how much does a repair cost", "what is the price of a tune up", "how much is a service"], "A basic tune-up costs $40."),
    (["do you rent bikes", "can i hire a bike", "how much is bike rental"], "Yes, bike rental is $15 per day."),
    (["can i return a bike", "what is your return policy", "can i get a refund"], "You can return items within 30 days with a receipt."),
    (["do you deliver", "can you deliver my bike", "do you offer delivery"], "We deliver within 10 km for $10."),
    (["how can i contact you", "what is your phone number", "how do i reach you"], "Call us at 555-0123 or email hello@example.com."),
    (["do you sell electric bikes", "do you have e-bikes", "how much is an electric bike"], "Yes, we sell electric bikes from $900."),
    (["what payment methods do you accept", "can i pay by card", "do you take cash"], "We accept cash, cards and mobile payments."),
]
rows = ["question,answer"] + [f'{q},"{a}"' for qs, a in faq for q in qs]
write("faq.csv", rows)

# science_notes.txt: a document for question answering and summarizing (from Solen's knowledge base)
import json
with open(os.path.join(HERE, "..", "knowledge.json")) as f:
    facts = [x["answer"] for x in json.load(f)["facts"] if x["topic"] == "science"]
write("science_notes.txt", [" ".join(facts[i:i + 4]) for i in range(0, len(facts), 4)])

# translations.csv: English -> Spanish sentence pairs (for translators)
nouns = [("cat", "gato", "m"), ("dog", "perro", "m"), ("house", "casa", "f"), ("car", "coche", "m"),
         ("boy", "niño", "m"), ("girl", "niña", "f"), ("book", "libro", "m"), ("table", "mesa", "f")]
adjs = [("big", "grande", "grande"), ("small", "pequeño", "pequeña"), ("red", "rojo", "roja"),
        ("new", "nuevo", "nueva"), ("old", "viejo", "vieja")]
rows = ["english,spanish"]
for en, es, g in nouns:
    art, una = ("el", "un") if g == "m" else ("la", "una")
    for a_en, a_m, a_f in adjs:
        rows.append(f"the {en} is {a_en},{art} {es} es {a_m if g == 'm' else a_f}")
    rows += [f"i see the {en},yo veo {art} {es}", f"the {en} is here,{art} {es} está aquí", f"i have a {en},yo tengo {una} {es}"]
write("translations.csv", rows)

# sounds/: short .wav clips of low tones, high tones and noise (for audio classifiers)
import struct
import wave
RATE = 8000
for kind in ("low_tone", "high_tone", "noise"):
    d = os.path.join(HERE, "sounds", kind)
    os.makedirs(d, exist_ok=True)
    for i in range(15):
        f0 = r.uniform(150, 300) if kind == "low_tone" else r.uniform(1200, 2000)
        frames = bytearray()
        for n in range(RATE // 2):
            v = r.uniform(-1, 1) if kind == "noise" else math.sin(2 * math.pi * f0 * n / RATE) + r.gauss(0, 0.05)
            frames += struct.pack("<h", int(max(-1, min(1, 0.6 * v)) * 32767))
        with wave.open(os.path.join(d, f"{kind}_{i:02d}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(bytes(frames))
print("examples written")
