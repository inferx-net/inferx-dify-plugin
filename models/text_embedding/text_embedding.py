import logging
import sys
import os
import time
from decimal import Decimal
from typing import Optional

import numpy as np
import requests
from dify_plugin import TextEmbeddingModel
from dify_plugin.entities.model import (
    AIModelEntity,
    EmbeddingInputType,
    FetchFrom,
    I18nObject,
    ModelPropertyKey,
    ModelType,
    PriceConfig,
    PriceType,
)
from dify_plugin.entities.model.text_embedding import EmbeddingUsage, TextEmbeddingResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from slots import get_embedding_url  # noqa: E402

logger = logging.getLogger(__name__)


class InferXEmbeddingModel(TextEmbeddingModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        endpoint = get_embedding_url(credentials["base_url"], model)
        headers = {"Content-Type": "application/json"}

        context_size = self._get_context_size(model, credentials)
        inputs = []
        for text in texts:
            num_tokens = self._get_num_tokens_by_gpt2(text)
            if num_tokens >= context_size:
                cutoff = int(np.floor(len(text) * (context_size / num_tokens)))
                inputs.append(text[:cutoff])
            else:
                inputs.append(text)

        payload = {"input": inputs, "model": model}

        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=(10, 300)
            )
        except requests.exceptions.ConnectTimeout as e:
            raise InvokeConnectionError(str(e))
        except requests.exceptions.ReadTimeout as e:
            raise InvokeConnectionError(str(e))
        except requests.exceptions.ConnectionError as e:
            raise InvokeConnectionError(str(e))

        if response.status_code == 401:
            raise InvokeAuthorizationError("InferX authentication failed (HTTP 401).")
        if response.status_code == 429:
            raise InvokeRateLimitError("InferX rate limit exceeded (HTTP 429).")
        if response.status_code >= 500:
            raise InvokeServerUnavailableError(
                f"InferX server error (HTTP {response.status_code}): {response.text}"
            )
        if response.status_code != 200:
            raise InvokeBadRequestError(
                f"InferX request failed (HTTP {response.status_code}): {response.text}"
            )

        data = response.json()
        # OpenAI-compatible: {"data": [{"embedding": [...], "index": 0}, ...], "usage": {...}}
        items = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in items]

        usage_data = data.get("usage", {})
        total_tokens = usage_data.get("total_tokens") or sum(
            self._get_num_tokens_by_gpt2(t) for t in inputs
        )
        usage = self._calc_embedding_usage(model, credentials, total_tokens)
        return TextEmbeddingResult(embeddings=embeddings, usage=usage, model=model)

    def get_num_tokens(self, model: str, credentials: dict, texts: list[str]) -> list[int]:
        return [self._get_num_tokens_by_gpt2(t) for t in texts]

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            self._invoke(model=model, credentials=credentials, texts=["ping"])
        except InvokeError as ex:
            raise CredentialsValidateFailedError(str(ex))
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.TEXT_EMBEDDING,
            fetch_from=FetchFrom.PREDEFINED_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: 8192,
                ModelPropertyKey.MAX_CHUNKS: 32,
            },
            parameter_rules=[],
            pricing=PriceConfig(
                input=Decimal("0"),
                unit=Decimal("0.000001"),
                currency="USD",
            ),
        )

    def _calc_embedding_usage(self, model: str, credentials: dict, tokens: int) -> EmbeddingUsage:
        input_price_info = self.get_price(
            model=model,
            credentials=credentials,
            price_type=PriceType.INPUT,
            tokens=tokens,
        )
        return EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=input_price_info.unit_price,
            price_unit=input_price_info.unit,
            total_price=input_price_info.total_amount,
            currency=input_price_info.currency,
            latency=time.perf_counter() - self.started_at,
        )

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeAuthorizationError: [requests.exceptions.InvalidHeader],
            InvokeBadRequestError: [
                requests.exceptions.HTTPError,
                requests.exceptions.InvalidURL,
            ],
            InvokeRateLimitError: [requests.exceptions.RetryError],
            InvokeServerUnavailableError: [requests.exceptions.ConnectionError],
            InvokeConnectionError: [
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
            ],
        }
