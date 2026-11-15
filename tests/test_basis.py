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
