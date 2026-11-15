# BuildYourModel: Solen 1.0

A small AI built from scratch in Python and NumPy. It has two models wired together:

| Part | File | What it does |
|---|---|---|
| **Brain** (neural network) | `ai/brain.py` | Understands the question. Hashed word, bigram and character features go into a sparse embedding layer (256 units), then a hidden layer (128 units), then a softmax over 590 classes (defined intents, knowledge facts and `unknown`). Trained with Adam, dropout, label smoothing and data augmentation (typos, filler words, word dropout). A TF-IDF retriever and spelling correction back it up. |
| **Language model** | `ai/language_model.py` | Its own trigram language model with interpolated Kneser-Ney smoothing. It scores text (perplexity), generates text, and holds the **defined responses** for every intent and fact. It ranks those responses by fluency and fills in slots such as `{name}`. |
| **Connector** | `ai/assistant.py` | Runs the pipeline: tools (name memory, calculator, `/teach`, `/generate`) → Brain picks the intent → the LM returns the defined response. |

The knowledge it is trained on lives in `data/knowledge.json` (479 facts). Its specialty is **building AI models**: 129 facts on the build process, data, training, architectures, LLM techniques, evaluation, tools, deployment and responsible AI, plus step-by-step guides and code examples. It also covers science, space, math, geography, history, computing, health, language and general knowledge. The defined conversation responses live in `data/intents.json`.

## Live site

**https://aiwebz.github.io/BuildYourModel/** runs Solen 1.0 entirely in the browser (Python + NumPy via Pyodide), so no server is needed. Built models are saved in the visitor's browser and their files never leave the device. The first visit downloads about 38 MB.

Rebuild and redeploy it with:

```bash
python tools/build_web.py site --standalone
```

then publish the `site/` folder to the `gh-pages` branch (or any static host).

## Website (Python server)

```bash
python server.py        # then open http://localhost:8000
```

The Solen 1.0 website lets anyone build an AI model in the browser: describe the model in the chat (or click an example), attach a CSV, TXT or ZIP of image folders, and Solen picks the model type, trains it and tests it. Every model appears in **Your models**, where you can run it and download it. Each browser gets its own private session (own chat memory, uploads and models). For safety the website only reads files the visitor uploaded plus the example data, and `/teach` is turned off there. It uses only the Python standard library plus NumPy (Pillow for image models).

## Run in the terminal

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

| Say something like | Model it builds |
|---|---|
| *make me an AI that predicts house prices from data/examples/houses.csv* | Number predictor (neural network) |
| *...with 3 hidden layers of 64 neurons, tanh, 400 epochs* | Number/table model with your own network design |
| *make a model that tells if a review is positive or negative* → *use the example data* | Text classifier |
| *build an AI that predicts which fruit it is from data/examples/fruit.csv* | Table classifier |
| *make a customer support chatbot from data/examples/faq.csv* | Chatbot (question → answer, says when it doesn't know) |
| *build an AI that answers questions about data/examples/science_notes.txt* | Document Q&A (BM25 passage retrieval) |
| *create a summarizer for data/examples/science_notes.txt* | Extractive summarizer (TextRank) |
| *make a translator using data/examples/translations.csv* | Translator (IBM Model 1 + language model, beam search) |
| *build a mini GPT transformer from data/examples/stories.txt* | Decoder-only transformer, hand-written backprop |
| *build a neural character model from data/examples/stories.txt* | Neural character-level language model |
| *make an AI that writes stories like data/examples/stories.txt* | Tiny n-gram language model |
| *train an AI to recognize shapes in data/examples/shapes* | Image classifier |
| *train an AI that generates images of shapes from data/examples/shapes* | Image generator (PCA + Gaussian latent model) |
| *make an AI that recognizes sounds in data/examples/sounds* | Audio classifier (.wav, FFT features) |
| *I want an AI that groups customers into 3 groups using data/examples/customers.csv* | Clustering (k-means) |
| *I need an AI that spots unusual transactions* → *use the example data* | Anomaly detector |
| *build a model that forecasts sales from data/examples/sales.csv* | Forecaster (autoregressive) |
| *create a recommender using data/examples/ratings.csv* | Recommender (collaborative filtering) |
| *make an AI agent that learns to escape the maze in data/examples/maze.txt* | Reinforcement learning agent (Q-learning) |

Requests at a scale that can't be trained here (ChatGPT-size language models, photorealistic image or video generators, speech synthesis, music generation) get an honest explanation, and Solen builds the closest small version where one exists.

Every model is sized to its data, scored on held-out data it never trained on (against a simple baseline where it makes sense), and saved. The example datasets are synthetic; regenerate them with `python data/examples/make_examples.py`.

Ask it things like *how do I build an AI model*, *how do I make my own LLM*, *how do I prevent overfitting*, *what is RAG* or *show me code for a neural network in pytorch*.

## Extend

Add entries to `data/knowledge.json` or `data/intents.json`, then run `python train.py` again.
