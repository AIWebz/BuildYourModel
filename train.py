"""Train both models (brain + language model) and run a quick evaluation."""
import time

from ai import Assistant

HELD_OUT = [  # phrasings never seen in training, with the expected label prefix
    ("yo whats good", "greeting"), ("catch you later", "goodbye"), ("thank u very much", "thanks"),
    ("whats the capitol of frnace", "fact"), ("how big is jupiter compared to other planets", "fact"),
    ("explain what a neurl network is", "fact"), ("who was the first guy on the moon", "fact"),
    ("how many hours should i sleep", "fact"), ("what are you exactly", "identity"),
    ("what stuff can you do", "capabilities"), ("tell me something funny", "joke"),
    ("book me a hotel in vegas", "__unknown__"),
]

if __name__ == "__main__":
    t = time.time()
    bot = Assistant()
    acc = bot.train(verbose=True)
    print(f"Done in {time.time() - t:.1f}s  (train accuracy {acc:.3f})\n")
    ok = 0
    for text, want in HELD_OUT:
        label, score, p, s = bot.analyze(text)[0]
        hit = label.startswith(want)
        ok += hit
        print(f"  [{'ok' if hit else '--'}] {text!r:50} -> {label:14} score {score:.2f}")
    print(f"\nHeld-out accuracy: {ok}/{len(HELD_OUT)}")
    sample = "Light travels at about 299,792 kilometers per second in a vacuum."
    print(f"LM perplexity on a known sentence: {bot.lm.perplexity(sample):.2f}")
    print(f"LM perplexity on word salad:       {bot.lm.perplexity('vacuum second light the per about'):.2f}")
