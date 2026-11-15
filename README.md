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
- **Describe the AI model you want in plain English** and it builds it: it works out the model type, the column to predict, the number of groups, the name and the data file. If something is missing, it asks and remembers your request.
- `/build name file [type]`: the same thing as a command
- `/ask name input`: runs a model it built
- `/models`: lists built models with parameter counts and file sizes (saved in `models/custom/`)

| Say something like | Model it builds | Then try |
|---|---|---|
| *make me an AI that predicts house prices from data/examples/houses.csv* | Number predictor (neural network) | `/ask houses 100, 3, downtown` |
| *make a model that tells if a review is positive or negative* → *use the example data* | Text classifier (neural network) | `/ask sentiment I love this` |
| *build an AI that predicts which fruit it is from data/examples/fruit.csv* | Table classifier (neural network) | `/ask fruit 90, yellow, bumpy` |
| *I want an AI that groups customers into 3 groups using data/examples/customers.csv* | Clustering (k-means, automatic k) | `/ask customers 25, 400, 9, app` |
| *I need an AI that spots unusual transactions* → *use the example data* | Anomaly detector | `/ask transactions 900, 3, 40, abroad` |
| *build a model that forecasts sales from data/examples/sales.csv* | Forecaster (autoregressive) | `/ask sales 6` |
| *create a recommender using data/examples/ratings.csv* | Recommender (collaborative filtering) | `/ask ratings space battle` |
| *build a neural character model called writer from data/examples/stories.txt* | Neural character-level language model | `/ask writer The robot` |
| *make an AI that writes stories like data/examples/stories.txt* | Tiny language model (Kneser-Ney trigram) | `/ask stories the bird` |
| *train an AI to recognize shapes in data/examples/shapes* | Image classifier (neural network, needs Pillow) | `/ask shapes data/examples/shapes/circle/circle_01.png` |
| *make an AI agent that learns to escape the maze in data/examples/maze.txt* | Reinforcement learning agent (Q-learning) | `/ask maze 3,8` |

Every model is sized to its data, scored on held-out data it never trained on (against a simple baseline where it makes sense), and saved. The example datasets are synthetic; regenerate them with `python data/examples/make_examples.py`.

Ask it things like *how do I build an AI model*, *how do I make my own LLM*, *how do I prevent overfitting*, *what is RAG* or *show me code for a neural network in pytorch*.

## Extend

Add entries to `data/knowledge.json` or `data/intents.json`, then run `python train.py` again.
