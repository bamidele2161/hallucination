"""Lynx-8B adapter (PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-v1.1).

Lynx reads the question, the document (``passage``) and the answer, writes its
reasoning, and ends with a verdict: ``{"REASONING": [...], "SCORE": "PASS"}``
for a faithful answer or ``"FAIL"`` for a hallucinated one. The prompt below is
copied verbatim from the model card and sent as a single user message, as the
card instructs.

Lynx gives no confidence, so its answer-level score is read from the model's
own token probabilities at the verdict:

    P(hallucinated) = P(FAIL) / (P(FAIL) + P(PASS))

The reply is generated greedily (up to 600 new tokens, as on the model card);
at the token where the verdict starts, the top-20 candidate tokens are split
into FAIL and PASS spellings. The verdict is conditioned on the model's own
reasoning, exactly as Lynx is meant to be used.

Training-data overlap: Lynx v1.1 was trained on CovidQA, PubMedQA and RAGTruth
data, so its results on ragbench_covidqa, ragbench_pubmedqa and ragtruth_qa
may be inflated; MedHallu is its only unseen dataset here.

Runs on vLLM (Linux + NVIDIA GPU); see vllm_engine.py.
"""

import re
from collections.abc import Sequence
from math import exp
from typing import Any

from .base import Detector

MODEL_ID = "PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-v1.1"
REVISION = "00de9f9062d93a95e4e359d3c92b8ea1d0114738"

PROMPT = """
Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}

--

Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
"""

TOP_LOGPROBS = 20  # vLLM's default maximum.
MAX_NEW_TOKENS = 600
# The verdict key, then the start of its value (an optional quote).
VERDICT_KEY = re.compile(r'"?SCORE"?\s*:\s*"?\s*', re.IGNORECASE)


def verdict_probability(
    tokens: Sequence[str], candidates: Sequence[Sequence[tuple[str, float]]]
) -> float | None:
    """P(FAIL) / (P(FAIL) + P(PASS)) at the verdict, or None if it cannot be read.

    ``tokens`` is the decoded text of each generated token; ``candidates[i]`` is
    the (decoded text, log-probability) list of the top tokens at position i.
    """
    text = "".join(tokens)
    starts = [
        match.end() for match in VERDICT_KEY.finditer(text)
        if text[match.end():match.end() + 4].upper() in ("PASS", "FAIL")
    ]
    if not starts:
        return None
    verdict_start = starts[-1]

    offset = 0
    for index, token in enumerate(tokens):
        if offset <= verdict_start < offset + len(token):
            break
        offset += len(token)
    else:
        return None
    # The part of this token before the verdict (e.g. ' "'); candidates must share it.
    lead = text[offset:verdict_start]

    mass = {"FAIL": 0.0, "PASS": 0.0}
    for candidate, logprob in candidates[index]:
        if not candidate.startswith(lead):
            continue
        tail = candidate[len(lead):].upper()
        for verdict in mass:
            if tail and (verdict.startswith(tail) or tail.startswith(verdict)):
                mass[verdict] += exp(logprob)
    total = mass["FAIL"] + mass["PASS"]
    return mass["FAIL"] / total if total > 0 else None


class LynxDetector(Detector):
    """Lynx-8B scored by its probability of answering FAIL."""

    name = "lynx-8b-instruct-v1.1"
    revision = REVISION

    def __init__(
        self,
        device: str | None = None,
        gpu_memory_utilization: float | str = 0.5,
        max_model_len: int | str = 32768,
        max_new_tokens: int | str = MAX_NEW_TOKENS,
    ) -> None:
        from vllm import SamplingParams

        from .vllm_engine import engine_options, load_llm, select_cuda_device

        select_cuda_device(device)
        self.llm = load_llm(
            MODEL_ID, REVISION, **engine_options(gpu_memory_utilization, max_model_len)
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(
            temperature=0, max_tokens=int(max_new_tokens), logprobs=TOP_LOGPROBS
        )
        self.unparsed: list[str] = []  # IDs whose verdict could not be read.
        self.last_replies: list[str] = []

    def prompt(self, example: dict[str, Any]) -> str:
        content = PROMPT.format(
            question=example["question"], context=example["passage"], answer=example["answer"]
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )

    def score_batch(self, batch: Sequence[dict[str, Any]]) -> list[float]:
        outputs = self.llm.generate(
            [self.prompt(example) for example in batch], self.sampling, use_tqdm=False
        )
        self.last_replies = [output.outputs[0].text for output in outputs]
        return [self._score(example, output.outputs[0]) for example, output in zip(batch, outputs)]

    def _score(self, example: dict[str, Any], completion: Any) -> float:
        tokens = [
            step[token_id].decoded_token or ""
            for token_id, step in zip(completion.token_ids, completion.logprobs)
        ]
        candidates = [
            [(entry.decoded_token or "", entry.logprob) for entry in step.values()]
            for step in completion.logprobs
        ]
        probability = verdict_probability(tokens, candidates)
        if probability is not None:
            return min(1.0, max(0.0, probability))
        # No readable verdict (e.g. the reply was cut off): count it and score as undecided.
        self.unparsed.append(example["id"])
        print(f"warning: lynx gave no PASS/FAIL verdict for id={example['id']}; scored 0.5")
        return 0.5


if __name__ == "__main__":
    # GPU smoke test: one supported and one hallucinated answer, with Lynx's reply.
    import sys

    from medical_hallucinations.datasets.base import read_jsonl

    dataset = sys.argv[1] if len(sys.argv) > 1 else "medhallu"
    rows = read_jsonl(f"data/processed/{dataset}.jsonl")
    examples = [next(row for row in rows if row["label"] == label) for label in (0, 1)]
    detector = LynxDetector()
    scores = detector.score_batch(examples)
    for example, score, reply in zip(examples, scores, detector.last_replies):
        print(f"id={example['id']} label={example['label']} score={score:.4f}")
        print(f"  reply: {reply}")
