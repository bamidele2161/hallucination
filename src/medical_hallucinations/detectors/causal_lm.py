"""Shared scoring for generative (causal LM) detectors such as LettuceDetect-v2 and Lynx.

Generative detectors answer in text, not with a probability. To get a
continuous score we use the model's own token probabilities: the probability
that, given the official prompt, the model would produce one exact reply
(e.g. ``{"hallucinated_spans": []}`` or ``"FAIL"``). This needs one forward
pass with the reply appended (teacher forcing), no sampling, and the result
is deterministic.

Examples are run one at a time so no padding is needed (padding interacts
badly with the recurrent layers in some recent architectures), and only the
logits of the reply positions are materialized, so long passages do not
allocate a full sequence x vocabulary tensor. Requires transformers >= 4.50
(the ``logits_to_keep`` forward argument).
"""

from collections.abc import Sequence
from typing import Any

from .base import resolve_device


class CompletionScorer:
    """Load a chat model once and score fixed replies to chat prompts."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str | None = None,
        dtype: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        import transformers

        self._torch = torch
        self.device = resolve_device(device)
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self.model = self._load_model(
            transformers, model_id, revision, getattr(torch, dtype), trust_remote_code
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _load_model(transformers: Any, model_id: str, revision: str, dtype: Any, remote: bool):
        """Prefer the text-only causal LM; fall back to the image-text class for VLM checkpoints."""
        errors = []
        for auto_class in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            loader = getattr(transformers, auto_class, None)
            if loader is None:
                continue
            # transformers 5 renamed torch_dtype to dtype.
            major = int(transformers.__version__.split(".")[0])
            dtype_key = "dtype" if major >= 5 else "torch_dtype"
            try:
                model, info = loader.from_pretrained(
                    model_id, revision=revision, trust_remote_code=remote,
                    output_loading_info=True, **{dtype_key: dtype},
                )
            except (ValueError, KeyError) as error:
                errors.append(f"{auto_class}: {error}")
                continue
            if info.get("missing_keys"):
                # Missing weights would be randomly initialized: never accept that.
                errors.append(f"{auto_class}: missing weights {info['missing_keys'][:5]}")
                continue
            return model
        raise RuntimeError(f"Could not load {model_id}: " + "; ".join(errors))

    def chat_prompt(self, messages: list[dict[str, str]], **template_options: Any) -> str:
        """Render messages with the model's chat template, ending at the assistant turn."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_options
        )

    def reply_logprobs(self, prompts: Sequence[str], replies: Sequence[str]) -> list[float]:
        """Return log P(reply | prompt) for each (rendered prompt, reply) pair."""
        if len(prompts) != len(replies):
            raise ValueError("prompts and replies must have the same length")
        return [self._reply_logprob(prompt, reply) for prompt, reply in zip(prompts, replies)]

    def _reply_logprob(self, prompt: str, reply: str) -> float:
        torch = self._torch
        # Generation continues from the prompt tokens, so the reply is tokenized
        # on its own and appended, exactly as the model would emit it.
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        reply_ids = self.tokenizer(reply, add_special_tokens=False)["input_ids"]
        if not reply_ids:
            raise ValueError("reply must contain at least one token")
        n_reply = len(reply_ids)

        input_ids = torch.tensor([prompt_ids + reply_ids], device=self.device)
        with torch.inference_mode():
            # Position t predicts token t + 1, so the reply's n tokens are
            # predicted by the n positions just before the end.
            logits = self.model(input_ids=input_ids, logits_to_keep=n_reply + 1).logits
        log_probs = torch.log_softmax(logits[0, :n_reply].float(), dim=-1)
        targets = input_ids[0, -n_reply:]
        return float(log_probs.gather(1, targets.unsqueeze(1)).sum())
