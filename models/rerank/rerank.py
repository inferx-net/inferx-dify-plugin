import json
import logging
import sys
import os
from typing import Optional

import requests
from dify_plugin import RerankModel
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    I18nObject,
    ModelPropertyKey,
    ModelType,
)
from dify_plugin.entities.model.rerank import RerankDocument, RerankResult
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
from slots import get_rerank_url  # noqa: E402

logger = logging.getLogger(__name__)


class InferXRerankModel(RerankModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        query: str,
        documents: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> RerankResult:
        if not documents:
            return RerankResult(model=model, docs=[])

        endpoint = get_rerank_url(credentials["base_url"], model)
        headers = {"Content-Type": "application/json"}

        payload: dict = {
            "model": model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

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
        # Expected response: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        results = data.get("results", [])
        if top_n is not None:
            results = results[:top_n]

        docs = []
        for item in results:
            index = item["index"]
            score = item.get("relevance_score", 0.0)
            if score_threshold is not None and score < score_threshold:
                continue
            # Some APIs nest the document text; fall back to original list
            text = item.get("document") or (documents[index] if index < len(documents) else "")
            if isinstance(text, dict):
                text = text.get("text", "")
            docs.append(RerankDocument(index=index, text=text, score=score))

        return RerankResult(model=model, docs=docs)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            self._invoke(
                model=model,
                credentials=credentials,
                query="What is the capital of France?",
                documents=["Paris is the capital of France.", "London is the capital of England."],
                score_threshold=0.0,
            )
        except InvokeError as ex:
            raise CredentialsValidateFailedError(str(ex))
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.RERANK,
            fetch_from=FetchFrom.PREDEFINED_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: 8192,
            },
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
