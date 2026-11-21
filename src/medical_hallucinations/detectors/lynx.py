"""TODO (Selma): Lynx-8B adapter (PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-v1.1).

How to plug into the shared pipeline (see Detector in base.py for the contract):

1. Subclass ``Detector``; set ``name`` and ``revision`` (pin the HF commit sha).
2. ``__init__(self, device=None, **options)``; load with
   ``causal_lm.CompletionScorer(MODEL_ID, REVISION, device=device)``.
3. Build the official Lynx prompt from ``example["question"]``,
   ``example["passage"]`` (the document) and ``example["answer"]``; copy it
   verbatim from the model card.
4. Lynx answers in JSON with its reasoning *before* ``"SCORE": "PASS"|"FAIL"``,
   so P(FAIL) is not the first token. One option: generate the reply greedily,
   cut it just before the score value, then compare
   ``scorer.reply_logprobs`` for the ``FAIL`` and ``PASS`` continuations and
   return P(FAIL) = e^f / (e^f + e^p). Check the exact format on the model card.
5. Register it in ``detectors/__init__.py`` as ``"lynx"``, add a unit test of
   the score conversion, then run
   ``medical-hallucinations infer --detector lynx --device cuda``.
"""
