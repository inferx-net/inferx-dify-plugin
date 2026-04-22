import json
import logging
import sys
import os
from collections.abc import Generator
from decimal import Decimal
from typing import Optional, Union, cast

import requests
from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
    PriceConfig,
)
from dify_plugin.entities.model.llm import (
    LLMMode,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel

# Add plugin root to path so we can import slots.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from slots import get_llm_url  # noqa: E402

logger = logging.getLogger(__name__)


class InferXLargeLanguageModel(LargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        return self._generate(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        return self._num_tokens_from_messages(prompt_messages)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            self._generate(
                model=model,
                credentials=credentials,
                prompt_messages=[UserPromptMessage(content="hi")],
                model_parameters={"max_tokens": 5},
                stream=False,
            )
        except InvokeError as ex:
            raise CredentialsValidateFailedError(str(ex))
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
    ) -> Union[LLMResult, Generator]:
        endpoint = get_llm_url(credentials["base_url"], model)
        headers = {"Content-Type": "application/json"}

        payload: dict = {
            "model": model,
            "messages": [self._message_to_dict(m) for m in prompt_messages],
            "stream": stream,
            **{k: v for k, v in model_parameters.items() if v is not None},
        }
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = [self._tool_to_dict(t) for t in tools]

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=(10, 300),
                stream=stream,
            )
        except requests.exceptions.ConnectTimeout as e:
            raise InvokeConnectionError(str(e))
        except requests.exceptions.ReadTimeout as e:
            raise InvokeConnectionError(str(e))
        except requests.exceptions.ConnectionError as e:
            raise InvokeConnectionError(str(e))

        if response.status_code == 401:
            raise InvokeAuthorizationError(
                f"InferX authentication failed (HTTP 401). Check base_url tenant token."
            )
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

        if stream:
            return self._handle_stream(model, credentials, prompt_messages, response)
        return self._handle_response(model, credentials, prompt_messages, response)

    def _handle_response(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        response: requests.Response,
    ) -> LLMResult:
        data = response.json()
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""

        tool_calls = []
        for raw_tc in msg.get("tool_calls") or []:
            tool_calls.append(self._parse_tool_call(raw_tc))

        assistant_message = AssistantPromptMessage(
            content=content, tool_calls=tool_calls
        )
        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", self._num_tokens_from_messages(prompt_messages))
        completion_tokens = usage_data.get("completion_tokens", self._get_num_tokens_by_gpt2(content))
        usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)
        return LLMResult(
            model=data.get("model", model),
            prompt_messages=prompt_messages,
            message=assistant_message,
            usage=usage,
        )

    def _handle_stream(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        response: requests.Response,
    ) -> Generator:
        full_content = ""
        chunk_index = 0
        # tool call accumulator: index -> AssistantPromptMessage.ToolCall
        tool_calls_by_index: dict[int, AssistantPromptMessage.ToolCall] = {}
        finish_reason = None

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason") or finish_reason

            content_piece = delta.get("content") or ""
            if content_piece:
                full_content += content_piece
                yield LLMResultChunk(
                    model=chunk.get("model", model),
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=chunk_index,
                        message=AssistantPromptMessage(content=content_piece),
                    ),
                )
                chunk_index += 1

            # Accumulate streaming tool calls
            for raw_tc in delta.get("tool_calls") or []:
                idx = raw_tc.get("index", 0)
                func = raw_tc.get("function", {})
                if idx not in tool_calls_by_index:
                    tool_calls_by_index[idx] = AssistantPromptMessage.ToolCall(
                        id=raw_tc.get("id", str(idx)),
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=func.get("name", ""),
                            arguments=func.get("arguments", ""),
                        ),
                    )
                else:
                    tool_calls_by_index[idx].function.arguments += func.get("arguments", "")

        # Final chunk with usage and finish_reason
        prompt_tokens = self._num_tokens_from_messages(prompt_messages)
        completion_tokens = self._get_num_tokens_by_gpt2(full_content)
        usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)

        final_tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
        yield LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(
                index=chunk_index,
                message=AssistantPromptMessage(content="", tool_calls=final_tool_calls),
                finish_reason=finish_reason or "stop",
                usage=usage,
            ),
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _message_to_dict(self, message: PromptMessage) -> dict:
        if isinstance(message, SystemPromptMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, UserPromptMessage):
            if isinstance(message.content, str):
                return {"role": "user", "content": message.content}
            # Multi-part (text + image)
            parts = []
            for part in message.content:
                if part.type == PromptMessageContentType.TEXT:
                    part = cast(TextPromptMessageContent, part)
                    parts.append({"type": "text", "text": part.data})
                elif part.type == PromptMessageContentType.IMAGE:
                    part = cast(ImagePromptMessageContent, part)
                    parts.append(
                        {"type": "image_url", "image_url": {"url": part.data}}
                    )
            return {"role": "user", "content": parts}
        if isinstance(message, AssistantPromptMessage):
            msg: dict = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            return msg
        if isinstance(message, ToolPromptMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        raise ValueError(f"Unknown message type: {type(message)}")

    def _tool_to_dict(self, tool: PromptMessageTool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _parse_tool_call(self, raw: dict) -> AssistantPromptMessage.ToolCall:
        func = raw.get("function", {})
        arguments = func.get("arguments", "")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        return AssistantPromptMessage.ToolCall(
            id=raw.get("id", ""),
            type="function",
            function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                name=func.get("name", ""),
                arguments=arguments,
            ),
        )

    def _num_tokens_from_messages(self, messages: list[PromptMessage]) -> int:
        total = 0
        for m in messages:
            if isinstance(m.content, str):
                total += self._get_num_tokens_by_gpt2(m.content)
            elif isinstance(m.content, list):
                for part in m.content:
                    if part.type == PromptMessageContentType.TEXT:
                        part = cast(TextPromptMessageContent, part)
                        total += self._get_num_tokens_by_gpt2(part.data)
        return total

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        # InferX uses predefined models only; this is a fallback stub.
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.LLM,
            fetch_from=FetchFrom.PREDEFINED_MODEL,
            model_properties={
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
                ModelPropertyKey.CONTEXT_SIZE: 32768,
            },
            parameter_rules=[
                ParameterRule(
                    name=DefaultParameterName.TEMPERATURE.value,
                    use_template=DefaultParameterName.TEMPERATURE.value,
                    label=I18nObject(en_US="Temperature"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name=DefaultParameterName.MAX_TOKENS.value,
                    use_template=DefaultParameterName.MAX_TOKENS.value,
                    label=I18nObject(en_US="Max Tokens"),
                    type=ParameterType.INT,
                    default=4096,
                    max=32768,
                ),
            ],
            features=[ModelFeature.TOOL_CALL, ModelFeature.STREAM_TOOL_CALL],
            pricing=PriceConfig(
                input=Decimal("0"),
                output=Decimal("0"),
                unit=Decimal("0.000001"),
                currency="USD",
            ),
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
