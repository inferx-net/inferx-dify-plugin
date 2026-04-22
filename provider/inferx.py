import logging

import requests
from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class InferXProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        Validate provider-level credentials by making a lightweight request
        to the LLM endpoint.  InferX authenticates via the URL itself (the
        tenant token is embedded in the base_url), so no separate API key
        is required.
        """
        base_url = credentials.get("base_url", "").rstrip("/")
        if not base_url:
            raise CredentialsValidateFailedError("base_url is required")

        # Quick reachability check: hit the /v1/models endpoint of the m1 slot
        # if available, or simply confirm the host responds.
        base = base_url[:-3] if base_url.endswith("/v1") else base_url
        probe_url = f"{base}/m1/v1/models"
        try:
            resp = requests.get(probe_url, timeout=10)
            # 401/403 means the server is reachable but rejected the tenant token
            if resp.status_code in (401, 403):
                raise CredentialsValidateFailedError(
                    f"InferX rejected the credentials (HTTP {resp.status_code}). "
                    "Check that the base_url contains a valid tenant token."
                )
        except CredentialsValidateFailedError:
            raise
        except requests.exceptions.ConnectionError as e:
            raise CredentialsValidateFailedError(
                f"Cannot connect to InferX endpoint '{probe_url}': {e}"
            )
        except requests.exceptions.Timeout:
            raise CredentialsValidateFailedError(
                f"Connection to InferX endpoint '{probe_url}' timed out"
            )
        except Exception as e:
            raise CredentialsValidateFailedError(
                f"Unexpected error validating InferX credentials: {e}"
            )
