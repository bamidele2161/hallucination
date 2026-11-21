"""Checks for the detector registry, device choice, and generative-score plumbing (no weights)."""
import unittest
from importlib import import_module
from importlib.util import find_spec
from math import exp, log
from unittest import mock

from medical_hallucinations.detectors import DETECTORS, load_detector
from medical_hallucinations.detectors.base import Detector, resolve_device


class RegistryTests(unittest.TestCase):
    def test_registered_adapters_resolve_to_detectors(self):
        for target in DETECTORS.values():
            module_name, class_name = target.split(":")
            cls = getattr(import_module(module_name), class_name)
            self.assertTrue(issubclass(cls, Detector))
            self.assertTrue(cls.name and cls.revision)

    def test_unknown_detector(self):
        with self.assertRaisesRegex(ValueError, "Unknown detector"):
            load_detector("nope")

    def test_explicit_device_passes_through(self):
        self.assertEqual(resolve_device("cuda:1"), "cuda:1")


class FakeScorer:
    def __init__(self, log_supported):
        self.log_supported = log_supported
        self.replies = []

    def chat_prompt(self, messages):
        return messages[-1]["content"]

    def reply_logprobs(self, prompts, replies):
        self.replies.extend(replies)
        return [self.log_supported[p] for p in prompts]


class LettuceDetectV2ScoreTests(unittest.TestCase):
    def test_score_is_one_minus_probability_of_empty_spans(self):
        from medical_hallucinations.detectors import lettucedetect

        detector = object.__new__(lettucedetect.LettuceDetectV2)
        detector.system_prompt = "system"
        detector.scorer = FakeScorer({"a": log(0.9), "b": log(0.2)})
        batch = [{"id": "a"}, {"id": "b"}]
        with mock.patch.object(lettucedetect, "_user_message", lambda e: e["id"]):
            scores = detector.score_batch(batch)
        self.assertAlmostEqual(scores[0], 0.1)
        self.assertAlmostEqual(scores[1], 0.8)
        self.assertEqual(detector.scorer.replies, [lettucedetect.SUPPORTED_REPLY] * 2)


@unittest.skipUnless(find_spec("torch"), "torch is not installed")
class CompletionScorerTests(unittest.TestCase):
    def test_reply_logprob_uses_positions_before_each_reply_token(self):
        import torch

        from medical_hallucinations.detectors.causal_lm import CompletionScorer

        vocab = 6
        generator = torch.Generator().manual_seed(0)
        full_logits = torch.randn(1, 5, vocab, generator=generator)

        class Tokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [int(ch) for ch in text]}

        class Model:
            def __call__(self, input_ids, logits_to_keep):
                assert input_ids.tolist() == [[1, 2, 3, 4, 5]]
                return mock.Mock(logits=full_logits[:, -logits_to_keep:])

        scorer = object.__new__(CompletionScorer)
        scorer._torch, scorer.device = torch, "cpu"
        scorer.tokenizer, scorer.model = Tokenizer(), Model()

        # Prompt "123", reply "45": token 4 is predicted at position 2, token 5 at position 3.
        log_probs = torch.log_softmax(full_logits[0], dim=-1)
        expected = float(log_probs[2, 4] + log_probs[3, 5])
        self.assertAlmostEqual(scorer.reply_logprobs(["123"], ["45"])[0], expected, places=5)
        self.assertLess(exp(expected), 1)


if __name__ == "__main__":
    unittest.main()
