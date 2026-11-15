import os
import shutil
import tempfile
import unittest

from ai import Assistant
from ai.assistant import calculate
from ai.language_model import LanguageModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BasisTest(unittest.TestCase):
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
        self.assertIn("Basis", self.bot.reply("who are you"))

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
