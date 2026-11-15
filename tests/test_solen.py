import os
import shutil
import tempfile
import unittest

from ai import Assistant
from ai.assistant import calculate
from ai.language_model import LanguageModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SolenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        data = os.path.join(cls.tmp, "data")
        shutil.copytree(os.path.join(ROOT, "data"), data,
                        ignore=shutil.ignore_patterns("learned.json"))
        cls.bot = Assistant(data_dir=data, model_dir=os.path.join(cls.tmp, "models"))
        cls.bot.train(verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_facts(self):
        self.assertIn("Paris", self.bot.reply("What's the capital of France?"))
        self.assertIn("Paris", self.bot.reply("whats the capitol of frnace"))
        self.assertIn("Jupiter", self.bot.reply("which planet is the biggest"))
        self.assertIn("206", self.bot.reply("how many bones does a human have"))

    def test_defined_responses(self):
        self.assertIn(self.bot.reply("hello"), [t.format(name_suffix="") for t in self.bot.lm.responses["greeting"]])
        self.assertIn("Solen 1.0", self.bot.reply("who are you"))

    def test_name_memory(self):
        self.assertIn("Riley", self.bot.reply("my name is riley"))
        self.assertIn("Riley", self.bot.reply("what is my name"))

    def test_calculator(self):
        self.assertEqual(calculate("what is 12 * 7")[1], 84)
        self.assertEqual(calculate("3 plus 4 times 2")[1], 11)
        self.assertIsNone(calculate("what is 2022"))
        self.assertIn("84", self.bot.reply("what is 12 times 7"))

    def test_unknown(self):
        self.assertIn("/teach", self.bot.reply("book me a hotel in vegas"))

    def test_teach(self):
        self.bot.reply("/teach what is the zorblax | The zorblax is a made-up test creature.")
        self.assertIn("zorblax", self.bot.reply("what is the zorblax"))

    def test_ai_building_knowledge(self):
        self.assertIn("baseline", self.bot.reply("how do i build an ai model"))
        self.assertIn("self-attention", self.bot.reply("explain how transformers work"))
        self.assertIn("tokenizer", self.bot.reply("how do i make my own llm"))
        self.assertIn("Kaggle", self.bot.reply("where do i get training data"))

    def test_build_small_models(self):
        ex = os.path.join(ROOT, "data", "examples")
        out = self.bot.reply(f"/build mood {ex}/sentiment.csv")
        self.assertIn("Built text classifier 'mood'", out)
        self.assertIn("positive", self.bot.reply("/ask mood i love it, it is amazing"))
        meta = self.bot.builder._meta("mood")
        self.assertLess(meta["bytes"], 2 * 1024 ** 2)  # truly small: under 2 MB

        self.assertIn("number predictor", self.bot.reply(f"/build homes {ex}/houses.csv"))
        price = float(self.bot.reply("/ask homes 100, 3, downtown").split(":")[-1].replace(",", ""))
        self.assertAlmostEqual(price, 325000, delta=30000)  # data formula gives 325,000
        self.assertIn("3 comma-separated", self.bot.reply("/ask homes 100"))

        self.assertIn("table classifier", self.bot.reply(f"/build fruits {ex}/fruit.csv"))
        self.assertIn("apple", self.bot.reply("/ask fruits 180, red, smooth"))

        self.assertIn("tiny language model", self.bot.reply(f"/build tales {ex}/stories.txt"))
        self.assertTrue(self.bot.reply("/ask tales the robot").startswith("[tales] The robot"))

        listing = self.bot.reply("/models")
        for n in ("mood", "homes", "fruits", "tales"):
            self.assertIn(n, listing)
        self.assertIn("no model named", self.bot.reply("/ask nothere hello"))
        self.assertIn("could not find", self.bot.reply("/build x missing.csv"))

        fresh = Assistant(data_dir=self.bot.data_dir, model_dir=self.bot.model_dir).load()
        self.assertIn("apple", fresh.reply("/ask fruits 200, green, smooth"))  # reload from disk

    def test_understands_model_requests(self):
        out = self.bot.reply("make me an ai that predicts house prices from data/examples/houses.csv")
        self.assertIn("number predictor", out)
        self.assertIn("Predicts price", out)
        self.assertIn("forecaster", self.bot.reply("build a model that forecasts sales from data/examples/sales.csv"))
        self.assertIn("next 3", self.bot.reply("/ask sales 3"))
        out = self.bot.reply("I want an AI that groups customers into 3 groups using data/examples/customers.csv")
        self.assertIn("Found 3 groups", out)
        out = self.bot.reply("make an ai agent that learns to escape the maze in data/examples/maze.txt")
        self.assertIn("route takes 11 steps (shortest possible: 11)", out)
        self.assertIn("recommender", self.bot.reply("create a model from data/examples/ratings.csv"))
        # multi-turn: ask for missing info, then use the example data
        self.assertIn("anomaly detector", self.bot.reply("I need an AI that spots unusual transactions"))
        self.assertIn("Built anomaly detector", self.bot.reply("use the example data"))
        self.assertIn("UNUSUAL", self.bot.reply("/ask transactions 900, 3, 40, abroad"))
        self.assertIn("What should the model do", self.bot.reply("build me an ai"))
        self.assertIn("table classifier", self.bot.reply("it should predict which fruit it is from data/examples/fruit.csv"))
        # knowledge questions are still answered, not turned into builds
        self.assertIn("pretrained", self.bot.reply("how do i build an image classifier"))

    def test_neural_and_image_models(self):
        out = self.bot.reply("build a neural character model called scribe from data/examples/stories.txt")
        self.assertIn("neural character model 'scribe'", out)
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")
        self.assertIn("image classifier", self.bot.reply("train an ai to recognize shapes in data/examples/shapes"))
        self.assertIn("triangle", self.bot.reply("/ask shapes data/examples/shapes/triangle/triangle_05.png"))

    def test_web_mode_restrictions(self):
        self.bot.allowed_roots = [os.path.join(ROOT, "data", "examples")]
        self.bot.allow_teach = False
        try:
            self.assertIn("only use files", self.bot.reply(f"/build leak {os.path.join(ROOT, 'README.md')}"))
            self.assertIn("only use files", self.bot.reply(
                f"make an ai that writes text like {os.path.join(ROOT, 'train.py')}.txt"))
            self.assertIn("turned off", self.bot.reply("/teach a | b"))
            self.assertIn("forecaster", self.bot.reply("build a model that forecasts sales from data/examples/sales.csv"))
        finally:
            self.bot.allowed_roots, self.bot.allow_teach = None, True

    def test_more_model_types(self):
        out = self.bot.reply("make a customer support chatbot from data/examples/faq.csv")
        self.assertIn("Built chatbot 'faq'", out)
        self.assertIn("repair", self.bot.reply("/ask faq can you fix my bike"))
        self.assertIn("don't know", self.bot.reply("/ask faq who won the world cup"))
        self.assertIn("document Q&A", self.bot.reply("build an ai that answers questions about data/examples/science_notes.txt"))
        self.assertIn("Gravity is", self.bot.reply("/ask science_notes what is gravity"))
        self.assertIn("could not find", self.bot.reply("/ask science_notes who won the world cup"))
        self.assertIn("summarizer", self.bot.reply("create a summarizer called digest for data/examples/science_notes.txt"))
        self.assertIn("translator", self.bot.reply("make a translator using data/examples/translations.csv"))
        self.assertEqual(self.bot.reply("/ask translations the girl is new"), "[translations] la niña es nueva")
        self.assertIn("image generator", self.bot.reply("train an ai that generates images of shapes from data/examples/shapes called artist"))
        self.assertIn("new triangle", self.bot.reply("/ask artist triangle"))
        self.assertIn("audio classifier", self.bot.reply("make an ai that recognizes sounds in data/examples/sounds"))
        self.assertIn("noise", self.bot.reply("/ask sounds data/examples/sounds/noise/noise_02.wav"))

    def test_mini_gpt(self):
        out = self.bot.reply("build a mini gpt transformer called tinygpt from data/examples/stories.txt")
        self.assertIn("decoder-only transformer", out)
        self.assertTrue(self.bot.reply("/ask tinygpt The robot").startswith("[tinygpt] The robot"))

    def test_custom_architecture_and_scope(self):
        out = self.bot.reply("build a model that predicts house prices from data/examples/houses.csv with 2 hidden "
                             "layers of 16 neurons, tanh, 300 epochs, called custom")
        self.assertIn("2 hidden layers (16, 16 units, tanh)", out)
        self.assertIn("thousands of GPUs", self.bot.reply("make me an AI like ChatGPT"))
        out = self.bot.reply("build an ai that generates videos of cats")
        self.assertIn("Video models", out)
        self.assertNotIn("Built", out)

    def test_save_load(self):
        fresh = Assistant(data_dir=self.bot.data_dir, model_dir=self.bot.model_dir).load()
        self.assertIn("Tokyo", fresh.reply("capital of japan"))

    def test_language_model(self):
        lm = self.bot.lm
        self.assertLess(lm.perplexity("The capital of France is Paris."),
                        lm.perplexity("Paris is France capital the of."))
        self.assertTrue(lm.generate("the capital", seed=1).startswith("The capital"))
        lm2 = LanguageModel().train(["a b c", "a b d"])
        total = sum(lm2.prob("a", "b", w) for w in lm2.vocab)
        self.assertAlmostEqual(total, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
