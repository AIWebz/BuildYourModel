# BuildYourModel: Basis

A small AI built from scratch in Python and NumPy. It has two models wired together:

| Part | File | What it does |
|---|---|---|
| **Brain** (neural network) | `ai/brain.py` | Understands the question. Hashed word, bigram and character features go into a sparse embedding layer (256 units), then a hidden layer (128 units), then a softmax over 460 classes (defined intents, knowledge facts and `unknown`). Trained with Adam, dropout, label smoothing and data augmentation (typos, filler words, word dropout). A TF-IDF retriever and spelling correction back it up. |
| **Language model** | `ai/language_model.py` | Its own trigram language model with interpolated Kneser-Ney smoothing. It scores text (perplexity), generates text, and holds the **defined responses** for every intent and fact. It ranks those responses by fluency and fills in slots such as `{name}`. |
| **Connector** | `ai/assistant.py` | Runs the pipeline: tools (name memory, calculator, `/teach`, `/generate`) → Brain picks the intent → the LM returns the defined response. |

The knowledge it is trained on lives in `data/knowledge.json` (350 basic facts covering science, space, nature, math, geography, history, computing, health, language and general knowledge). The defined conversation responses live in `data/intents.json`.

## Run

```bash
pip install numpy
python train.py        # trains both models (~25s) and prints an evaluation
python chat.py         # chat; add --debug to see what the brain predicted
python -m unittest tests.test_basis -v
```

In chat:
- `my name is Sam`: it remembers your name
- `what is 12 times 7`: built-in calculator
- `/teach what is X | X is ...`: teaches it a new fact (saved to `data/learned.json`)
- `/generate the capital`: free text from the language model

## Extend

Add entries to `data/knowledge.json` or `data/intents.json`, then run `python train.py` again.
