"""Checks for the detector registry, device choice, and generative-score plumbing (no weights)."""
import unittest
from importlib import import_module
from importlib.util import find_spec
import sys
from math import exp, log
from types import ModuleType, SimpleNamespace
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


class LynxVerdictTests(unittest.TestCase):
    def setUp(self):
        from medical_hallucinations.detectors.lynx import verdict_probability

        self.verdict_probability = verdict_probability

    def test_reads_fail_and_pass_mass_at_the_verdict_token(self):
        tokens = ['{"', "RE", "ASON", 'ING": ["ok"], "', "SCORE", '":', ' "', "FAIL", '"}']
        candidates = [[(t, 0.0)] for t in tokens]
        candidates[7] = [("FAIL", log(0.6)), ("PASS", log(0.3)), ("F", log(0.05)), ("x", log(0.05))]
        self.assertAlmostEqual(self.verdict_probability(tokens, candidates), 0.65 / 0.95)

    def test_verdict_inside_a_token_matches_candidates_with_the_same_lead(self):
        tokens = ['{"REASONING": [], "SCORE":', ' "PASS', '"}']
        candidates = [[], [(' "PASS', log(0.8)), (' "FAIL', log(0.2)), (' "', log(0.1))], []]
        self.assertAlmostEqual(self.verdict_probability(tokens, candidates), 0.2)

    def test_uses_the_last_score_key_and_handles_missing_verdicts(self):
        tokens = ['"SCORE" is PASS or FAIL.', ' "SCORE": "', "PASS"]
        candidates = [[], [], [("PASS", log(0.5)), ("FAIL", log(0.5))]]
        self.assertAlmostEqual(self.verdict_probability(tokens, candidates), 0.5)
        self.assertIsNone(self.verdict_probability(['{"REASONING": ["cut off'], [[]]))

    def test_unreadable_verdict_scores_half_and_is_recorded(self):
        from medical_hallucinations.detectors.lynx import LynxDetector

        detector = object.__new__(LynxDetector)
        detector.unparsed = []
        step = {7: SimpleNamespace(decoded_token="no verdict", logprob=0.0)}
        completion = SimpleNamespace(token_ids=[7], logprobs=[step])
        with mock.patch("builtins.print"):
            self.assertEqual(detector._score({"id": "x"}, completion), 0.5)
        self.assertEqual(detector.unparsed, ["x"])


class MiniCheckScoreTests(unittest.TestCase):
    def test_score_is_one_minus_support_and_uses_passage_and_answer(self):
        from medical_hallucinations.detectors.minicheck import MiniCheckDetector

        calls = {}

        class Scorer:
            def score(self, docs, claims):
                calls.update(docs=docs, claims=claims)
                return [1, 0], [0.9, 0.25], None, None

        detector = object.__new__(MiniCheckDetector)
        detector.scorer = Scorer()
        batch = [{"passage": "P1", "answer": "A1", "question": "Q"},
                 {"passage": "P2", "answer": "A2", "question": "Q"}]
        scores = detector.score_batch(batch)
        self.assertAlmostEqual(scores[0], 0.1)
        self.assertAlmostEqual(scores[1], 0.75)
        self.assertEqual(calls, {"docs": ["P1", "P2"], "claims": ["A1", "A2"]})


class VllmEngineTests(unittest.TestCase):
    def test_pinned_vllm_injects_revision_only_for_the_expected_model(self):
        from medical_hallucinations.detectors.vllm_engine import pinned_vllm

        fake = ModuleType("vllm")
        fake.LLM = original = lambda **kwargs: kwargs
        with mock.patch.dict(sys.modules, {"vllm": fake}):
            with pinned_vllm("org/model", "abc", gpu_memory_utilization=0.4):
                import vllm

                built = vllm.LLM(model="org/model", seed=2024)
                with self.assertRaisesRegex(ValueError, "Expected"):
                    vllm.LLM(model="other/model")
            self.assertIs(fake.LLM, original)  # Restored after the block.
        self.assertEqual(built, {"model": "org/model", "seed": 2024, "revision": "abc",
                                 "tokenizer_revision": "abc", "gpu_memory_utilization": 0.4})

    def test_device_selection(self):
        from medical_hallucinations.detectors.vllm_engine import select_cuda_device

        with mock.patch.dict("os.environ", {}, clear=False):
            select_cuda_device("cuda:2")
            import os

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
        with self.assertRaisesRegex(ValueError, "NVIDIA"):
            select_cuda_device("cpu")


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
