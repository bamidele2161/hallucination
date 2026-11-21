"""LettuceDetect adapters: the v2 generative span detector and the v1 token classifier.

Both use the official ``lettucedetect`` package for prompt construction, so the
input matches what the models were trained on. Each answer is checked against
``[passage]`` and its ``question`` with the package's QA prompt template.

v2 (``lettucedect-v2-qwen-2b``, the default) replies with JSON listing
hallucinated spans and gives no confidence. Its answer-level score is

    P(hallucinated) = 1 - P(reply == '{"hallucinated_spans": []}' then end of turn)

computed from the model's token probabilities by teacher forcing (see
causal_lm.py). An answer the model would flag with any span therefore scores
high, and the score is continuous, as AUROC and calibration require.

v1 (``lettucedect-large-modernbert-en-v1``) is a token classifier. Its
answer-level score is the maximum hallucination probability over answer tokens,
the rule LettuceDetect uses for example-level decisions.

Both models were trained on RAGTruth train data, which is why only RAGTruth
test is used in this study.
"""

from collections.abc import Sequence
from math import exp
from typing import Any

from .base import Detector, resolve_device

V2_MODEL_ID = "KRLabsOrg/lettucedect-v2-qwen-2b"
V2_REVISION = "1833c76f4408f8e3e36a62896793348f3cbfb62e"
V1_MODEL_ID = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
V1_REVISION = "22296c700ef0ba4ab3e5c9afffa0185caaf61e52"

# The model's trained reply for a fully supported answer, then its end-of-turn token.
SUPPORTED_REPLY = '{"hallucinated_spans": []}<|im_end|>'


def _user_message(example: dict[str, Any]) -> str:
    from lettucedetect.detectors.prompt_utils import PromptUtils
    from lettucedetect.prompts.generative import build_user_message

    context = PromptUtils.format_context([example["passage"]], example["question"], "en")
    return build_user_message(context, example["answer"])


class LettuceDetectV2(Detector):
    """Generative span detector scored by P(no hallucinated spans)."""

    name = "lettucedetect-v2-qwen-2b"
    revision = V2_REVISION

    def __init__(self, device: str | None = None, dtype: str | None = None) -> None:
        from lettucedetect.prompts.generative import SYSTEM_BASE

        from .causal_lm import CompletionScorer

        self.system_prompt = SYSTEM_BASE
        self.scorer = CompletionScorer(V2_MODEL_ID, V2_REVISION, device=device, dtype=dtype)

    def prompt(self, example: dict[str, Any]) -> str:
        """The rendered chat prompt the model sees for one example."""
        return self.scorer.chat_prompt([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": _user_message(example)},
        ])

    def score_batch(self, batch: Sequence[dict[str, Any]]) -> list[float]:
        prompts = [self.prompt(example) for example in batch]
        log_supported = self.scorer.reply_logprobs(prompts, [SUPPORTED_REPLY] * len(prompts))
        return [min(1.0, max(0.0, 1.0 - exp(value))) for value in log_supported]

    def generate(self, example: dict[str, Any], max_new_tokens: int = 256) -> str:
        """Greedy reply, for checking that scores agree with the model's own verdict."""
        torch = self.scorer._torch
        tokenizer = self.scorer.tokenizer
        ids = tokenizer(self.prompt(example), add_special_tokens=False, return_tensors="pt")
        ids = {key: value.to(self.scorer.device) for key, value in ids.items()}
        with torch.inference_mode():
            output = self.scorer.model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=False
            )
        return tokenizer.decode(output[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


class LettuceDetectV1(Detector):
    """ModernBERT token classifier scored by the maximum token hallucination probability."""

    name = "lettucedetect-large-modernbert-en-v1"
    revision = V1_REVISION

    def __init__(self, device: str | None = None, max_length: int | str = 4096) -> None:
        from lettucedetect.models.inference import HallucinationDetector

        self.detector = HallucinationDetector(
            method="transformer",
            model_path=V1_MODEL_ID,
            revision=V1_REVISION,
            device=resolve_device(device),
            max_length=int(max_length),
        )

    def score_batch(self, batch: Sequence[dict[str, Any]]) -> list[float]:
        scores = []
        for example in batch:
            tokens = self.detector.predict(
                context=[example["passage"]],
                question=example["question"],
                answer=example["answer"],
                output_format="tokens",
            )
            scores.append(max((token["prob"] for token in tokens), default=0.0))
        return scores


if __name__ == "__main__":
    # GPU smoke test: compare the score with the model's own greedy verdict.
    import sys

    from medical_hallucinations.datasets.base import read_jsonl

    dataset = sys.argv[1] if len(sys.argv) > 1 else "ragtruth_qa"
    rows = read_jsonl(f"data/processed/{dataset}.jsonl")
    examples = [next(row for row in rows if row["label"] == label) for label in (0, 1)]
    detector = LettuceDetectV2()
    for example, score in zip(examples, detector.score_batch(examples)):
        print(f"id={example['id']} label={example['label']} score={score:.4f}")
        print(f"  greedy reply: {detector.generate(example)}")
