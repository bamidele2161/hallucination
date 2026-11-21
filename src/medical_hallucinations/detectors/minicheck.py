"""TODO (Selma): Bespoke-MiniCheck-7B adapter (bespokelabs/Bespoke-MiniCheck-7B).

How to plug into the shared pipeline (see Detector in base.py for the contract):

1. Subclass ``Detector``; set ``name`` and ``revision`` (pin the HF commit sha).
2. ``__init__(self, device=None, **options)``; load the model through the
   official ``minicheck`` package (see the model card for install steps).
3. MiniCheck checks a claim against a document: pass ``example["passage"]`` as
   the document and ``example["answer"]`` as the claim. It does not take the
   question.
4. MiniCheck's probability means *supported*, so return
   ``1 - P(supported)`` as P(hallucinated).
5. Register it in ``detectors/__init__.py`` as ``"minicheck"``, add a unit
   test of the score conversion, then run
   ``medical-hallucinations infer --detector minicheck --device cuda``.
"""
