# BuildYourModel: Basis

A small AI built from scratch in Python and NumPy. It has two models wired together:

| Part | File | What it does |
|---|---|---|
| **Brain** (neural network) | `ai/brain.py` | Understands the question. Hashed word, bigram and character features go into a sparse embedding layer (256 units), then a hidden layer (128 units), then a softmax over 590 classes (defined intents, knowledge facts and `unknown`). Trained with Adam, dropout, label smoothing and data augmentation (typos, filler words, word dropout). A TF-IDF retriever and spelling correction back it up. |
| **Language model** | `ai/language_model.py` | Its own trigram language model with interpolated Kneser-Ney smoothing. It scores text (perplexity), generates text, and holds the **defined responses** for every intent and fact. It ranks those responses by fluency and fills in slots such as `{name}`. |
| **Connector** | `ai/assistant.py` | Runs the pipeline: tools (name memory, calculator, `/teach`, `/generate`) → Brain picks the intent → the LM returns the defined response. |

The knowledge it is trained on lives in `data/knowledge.json` (479 facts). Its specialty is **building AI models**: 129 facts on the build process, data, training, architectures, LLM techniques, evaluation, tools, deployment and responsible AI, plus step-by-step guides and code examples. It also covers science, space, math, geography, history, computing, health, language and general knowledge. The defined conversation responses live in `data/intents.json`.

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
- `/build name file`: **builds and trains a new small AI model** from your file, sized to the data, then reports held-out accuracy, parameter count and file size
- `/ask name input`: runs a model it built
- `/models`: lists built models (saved in `models/custom/`)

| File you give it | Model it builds | Example (files in `data/examples/`) |
|---|---|---|
| `.csv` with `text,label` | Text classifier (small neural network) | `/build sentiment data/examples/sentiment.csv` → `/ask sentiment I love this` |
| `.csv` table, last column a number | Number predictor (regression neural network) | `/build houses data/examples/houses.csv` → `/ask houses 100, 3, downtown` |
| `.csv` table, last column a category | Table classifier (neural network) | `/build fruit data/examples/fruit.csv` → `/ask fruit 90, yellow, bumpy` |
| `.txt` plain text | Tiny language model (Kneser-Ney trigram) | `/build robot data/examples/stories.txt` → `/ask robot the bird` |

`houses.csv` and `fruit.csv` are synthetic data generated for the demos.

Ask it things like *how do I build an AI model*, *how do I make my own LLM*, *how do I prevent overfitting*, *what is RAG* or *show me code for a neural network in pytorch*.

## Extend

Add entries to `data/knowledge.json` or `data/intents.json`, then run `python train.py` again.
