"""Biod: zero-training decision model for tool-library / knowledge-base selection.

Package layout
--------------
catalog  : parse bioagent-dev's on-disk ``toollibrary/`` and ``knowledgebase/`` trees
render   : render <state> + <options> prompts for the two decision types
backend  : model backends (HF transformers / deterministic mock)
readout  : logit readouts (letter choice / pointwise yes-no)
decide   : hierarchical + flat decision engine with thresholds
replay   : Dream-RSI style replay simulator over recorded decisions
cli      : command line entry points
"""

__version__ = "0.1.0"
