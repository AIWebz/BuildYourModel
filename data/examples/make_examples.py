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
print("examples written")
