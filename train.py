"""Train both models (brain + language model) and run a quick evaluation."""
import time

from ai import Assistant

HELD_OUT = [  # phrasings never seen in training -> text the reply must contain
    ("yo whats good", "Hi|Hello|Hey"), ("catch you later", "Bye|Goodbye|See you"),
    ("thank u very much", "welcome|help|Any time"), ("whats the capitol of frnace", "Paris"),
    ("explain what a neurl network is", "layers"), ("who was the first guy on the moon", "Armstrong"),
    ("how many hours should i sleep", "7 to 9"), ("what are you exactly", "Solen 1.0"),
    ("what stuff can you do", "science"), ("tell me something funny", "?|computer"),
    ("book me a hotel in vegas", "/teach"), ("why is the sky colored blue", "scatter"),
    ("whats the capital city of egypt", "Cairo"), ("who came up with the theory of evolution", "Darwin"),
    ("how do i make a strong pasword", "passphrase"), ("how many teeth does a grown up have", "32"),
    ("what is a haiku poem", "5, 7"), ("how tall is a giraffe", "giraffe"),
    ("explain black hole", "gravity"), ("who was the first human to go to space", "Gagarin"),
    ("what is the hottest planet in our solar system", "Venus"), ("how big is jupiter compared to other planets", "Jupiter"),
    ("who started microsoft", "Gates"), ("whats the biggest ocean on earth", "Pacific"),
    ("what is the meaning of life", "/teach"), ("how are rainbows made", "sunlight"),
    ("what steps do i follow to make an ai model", "baseline"), ("my model memorizes the training set", "Overfitting|overfitting"),
    ("explain how transformers work", "self-attention"), ("how do i make my own llm", "tokenizer"),
    ("which python libraries should i use for deep learning", "PyTorch"), ("how do i pick a learnin rate", "0.001"),
    ("whats the difference between precision and recall", "Precision"), ("how do i give a chatbot my company documents", "retriev"),
    ("how can i shrink a model so it runs faster", "bits|student|Pruning"), ("where do i get training data", "Kaggle"),
]

if __name__ == "__main__":
    t = time.time()
    bot = Assistant()
    acc = bot.train(verbose=True)
    print(f"Done in {time.time() - t:.1f}s  (train accuracy {acc:.3f})\n")
    ok = 0
    for text, want in HELD_OUT:
        reply = bot.reply(text)
        hit = any(w in reply for w in want.split("|"))
        ok += hit
        print(f"  [{'ok' if hit else '--'}] {text!r:48} -> {reply[:60]}")
    print(f"\nHeld-out accuracy: {ok}/{len(HELD_OUT)}")
    sample = "Light travels at about 299,792 kilometers per second in a vacuum."
    print(f"LM perplexity on a known sentence: {bot.lm.perplexity(sample):.2f}")
    print(f"LM perplexity on word salad:       {bot.lm.perplexity('vacuum second light the per about'):.2f}")
