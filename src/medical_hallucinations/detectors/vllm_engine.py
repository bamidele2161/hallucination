"""Shared vLLM loading for the 7-8B generative detectors (Lynx, MiniCheck).

vLLM batches generation on the GPU, which makes scoring thousands of answers
with an 8B model take minutes instead of hours. It needs Linux and an NVIDIA
GPU; install it in its own environment (the ``vllm`` extra), because it pins
its own torch and transformers versions.

Every engine is pinned to a model commit (``revision``) so cached scores never
mix model versions, and uses greedy decoding with a fixed seed.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

DEFAULT_GPU_MEMORY = 0.5  # Fraction of GPU memory vLLM reserves; plenty for 8B on large GPUs.


def select_cuda_device(device: str | None) -> None:
    """Map the CLI ``--device`` onto vLLM, which only runs on CUDA.

    ``cuda:N`` sets CUDA_VISIBLE_DEVICES, so it must run before vLLM starts.
    """
    if device in (None, "auto", "cuda"):
        return
    if device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        return
    raise ValueError(f"vLLM detectors need an NVIDIA GPU (--device cuda), got {device!r}")


def engine_options(
    gpu_memory_utilization: float | str = DEFAULT_GPU_MEMORY,
    max_model_len: int | str | None = None,
) -> dict[str, Any]:
    """vLLM constructor options from CLI strings (``--option key=value``)."""
    options: dict[str, Any] = {"gpu_memory_utilization": float(gpu_memory_utilization), "seed": 0}
    if max_model_len is not None:
        options["max_model_len"] = int(max_model_len)
    return options


def load_llm(model_id: str, revision: str, **options: Any) -> Any:
    """A vLLM engine for ``model_id`` at the pinned ``revision``."""
    from vllm import LLM

    return LLM(model=model_id, revision=revision, tokenizer_revision=revision, **options)


@contextmanager
def pinned_vllm(model_id: str, revision: str, **options: Any) -> Iterator[None]:
    """Make third-party code that builds ``vllm.LLM(model=model_id)`` load our pinned revision.

    The official MiniCheck package creates its own engine without a revision
    argument; this injects ``revision`` and ``options`` for that one call.
    """
    import vllm

    original = vllm.LLM

    def build(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", args[0] if args else None)
        if model != model_id:
            raise ValueError(f"Expected vLLM to load {model_id}, got {model}")
        for key, value in options.items():
            kwargs.setdefault(key, value)
        return original(*args, revision=revision, tokenizer_revision=revision, **kwargs)

    vllm.LLM = build
    try:
        yield
    finally:
        vllm.LLM = original
