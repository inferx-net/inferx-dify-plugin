"""
InferX slot configuration.

The key concept is a "func name" — the deployment identifier embedded in the
base URL the user provides.  For example:

  https://model.inferx.net/funccall/tn-fklu9jum6b/default/Qwen3.6-35B-A3B-FP8/v1
                                                           ^^^^^^^^^^^^^^^^^^^^
                                                           func name

Each func name deployment maps to its own set of models per slot:

  FUNC_MODELS[func_name][slot] = [list of model names served at that slot]

URL construction rules (base = base_url with trailing "/v1" stripped):

  LLM    → {base}/{slot}/v1/chat/completions
  Embed  → {base}/{slot}/v1/embeddings
  Rerank → {base}/{slot}/rerank

To add a new func deployment, add an entry to FUNC_MODELS.  The rest of the
code derives everything from this dict automatically.
"""

# func_name -> slot_id -> list of model names
FUNC_MODELS: dict[str, dict[str, list[str]]] = {
    "Qwen3.6-35B-A3B-FP8": {
        "m1": ["Qwen/Qwen3.6-35B-A3B-FP8"],
        "m2": ["Qwen/Qwen3-Reranker-0.6B"],
        "m3": ["Qwen/Qwen3-Embedding-0.6B"],
    },
    # Example of a second func deployment with a different model set:
    # "func2": {
    #     "m1": ["Qwen/Qwen3.6-35B-A3B"],
    #     "m2": ["Qwen/Qwen3-Embedding-8B"],
    #     "m3": ["Qwen/Qwen3-Reranker-6B"],
    # },
}


def _strip_v1(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _extract_func_name(base_url: str) -> str:
    """Return the last path segment of base_url after stripping a trailing /v1.

    e.g. https://host/funccall/tn-xxx/default/Qwen3.6-35B-A3B-FP8/v1
         → "Qwen3.6-35B-A3B-FP8"
    """
    return _strip_v1(base_url).rstrip("/").rsplit("/", 1)[-1]


def _find_slot(base_url: str, model: str, fallback: str) -> str:
    """Look up which slot serves *model* for the func deployment in base_url."""
    func_name = _extract_func_name(base_url)
    slot_map = FUNC_MODELS.get(func_name, {})
    for slot, models in slot_map.items():
        if model in models:
            return slot
    return fallback


def get_llm_url(base_url: str, model: str) -> str:
    slot = _find_slot(base_url, model, fallback="m1")
    return f"{_strip_v1(base_url)}/{slot}/v1/chat/completions"


def get_embedding_url(base_url: str, model: str) -> str:
    slot = _find_slot(base_url, model, fallback="m2")
    return f"{_strip_v1(base_url)}/{slot}/v1/embeddings"


def get_rerank_url(base_url: str, model: str) -> str:
    slot = _find_slot(base_url, model, fallback="m3")
    return f"{_strip_v1(base_url)}/{slot}/rerank"
