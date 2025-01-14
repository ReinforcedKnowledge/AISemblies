from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aisemblies.tools import ToolCall


@dataclass
class UsageBreakdown:
    """
    Breakdown of token usage.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


@dataclass
class AssistantResponseMessage:
    """
    Utility class to model the assistant's response as a message it can receive back.
    """

    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class AssistantChoice:
    """
    One choice from the response, containing the assistant's final message
    and why the model stopped generating further tokens.
    """

    message: AssistantResponseMessage
    finish_reason: str
    index: int


@dataclass
class AssistantResponse:
    """
    Wrapper around /chat/completions response data.
    """

    id: str
    model: str
    created: int
    choices: list[AssistantChoice]
    usage: UsageBreakdown | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    object_type: str | None = None

    @property
    def first_choice(self) -> AssistantChoice | None:
        """
        Get the first choice from the response if available.

        Returns
        -------
        AssistantChoice | None
            The first AssistantChoice or `None` if no choices exist.
        """
        return self.choices[0] if self.choices else None

    def get_choice(self, choice_index) -> AssistantChoice:
        """
        Retrieve a specific choice by its index.

        Parameters
        ----------
        choice_index : int
            The index of the choice to retrieve.

        Returns
        -------
        AssistantChoice
            The AssistantChoice at the specified index.
        """
        return self.choices[choice_index]

    def get_all_choices(self) -> list[AssistantChoice]:
        """
        Retrieve all choices from the response.

        Returns
        -------
        list of AssistantChoice
            All AssistantChoice instances in the response.
        """
        return self.choices

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        """
        Get all tool calls from all choices in the response.

        Returns
        -------
        list of ToolCall
            All ToolCall instances found across choices.
        """
        calls: list[ToolCall] = []
        for choice in self.choices:
            if choice.message.tool_calls:
                calls.extend(choice.message.tool_calls)
        return calls

    @classmethod
    def from_completion(cls, data: Any) -> AssistantResponse:
        """
        Create an `AssistantResponse` from the raw result of a `/chat/completions` call.

        Parameters
        ----------
        data : Any
            The raw completion object.

        Returns
        -------
        AssistantResponse
            Parsed `AssistantResponse` containing model, usage, and choice data.
        """
        object_type = getattr(data, "object", None)
        resp_id = getattr(data, "id", "")
        model = getattr(data, "model", "")
        created = getattr(data, "created", 0)
        system_fp = getattr(data, "system_fingerprint", None)
        service_tier = getattr(data, "service_tier", None)
        usage_data = getattr(data, "usage", None)
        usage = cls._parse_usage(usage_data)

        raw_choices = getattr(data, "choices", [])
        parsed_choices: list[AssistantChoice] = []

        for choice in raw_choices:
            finish_reason = getattr(choice, "finish_reason", None)
            index = getattr(choice, "index", 0)
            if object_type == "chat.completion.chunk":
                delta = getattr(choice, "delta", None)
                msg_data = cls._delta_to_message(delta)
            else:
                msg_data = getattr(choice, "message", None)

            if not msg_data:
                assistant_msg = AssistantResponseMessage()
            else:
                content = getattr(msg_data, "content", None)
                refusal = getattr(msg_data, "refusal", None)
                tool_calls = getattr(msg_data, "tool_calls", None)
                tool_calls_parsed = cls._parse_tool_calls(tool_calls)
                assistant_msg = AssistantResponseMessage(
                    content=content,
                    refusal=refusal,
                    tool_calls=tool_calls_parsed,
                )

            c = AssistantChoice(
                message=assistant_msg,
                finish_reason=finish_reason,
                index=index,
            )
            parsed_choices.append(c)

        return cls(
            id=resp_id,
            model=model,
            created=created,
            choices=parsed_choices,
            usage=usage,
            system_fingerprint=system_fp,
            service_tier=service_tier,
            object_type=object_type,
        )

    @staticmethod
    def _delta_to_message(delta: Any) -> AssistantResponseMessage:
        """
        Convert a streaming chunk's `choice.delta` to an `AssistantResponseMessage`.

        Parameters
        ----------
        delta : Any
            The streaming delta object from the API.

        Returns
        -------
        AssistantResponseMessage
            A partial or complete message from a streaming chunk.
        """
        if not delta:
            return AssistantResponseMessage()
        content = getattr(delta, "content", None)
        refusal = getattr(delta, "refusal", None)
        tool_calls = getattr(delta, "tool_calls", None)
        tool_calls_parsed = AssistantResponse._parse_tool_calls(tool_calls)
        return AssistantResponseMessage(
            content=content, refusal=refusal, tool_calls=tool_calls_parsed
        )

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall] | None:
        """
        Convert the model's `tool_calls` data into a list of `ToolCall`.

        Parameters
        ----------
        raw_tool_calls : Any
            Data from the API representing tool calls.

        Returns
        -------
        list[ToolCall] | None
            List of parsed `ToolCall` objects or `None` if none provided.
        """
        if not raw_tool_calls:
            return None
        calls: list[ToolCall] = []
        for call in raw_tool_calls:
            call_id = getattr(call, "id", "")
            call_type = getattr(call, "type", "function")
            func_obj = getattr(call, "function", None)
            name = getattr(func_obj, "name", "") if func_obj else ""
            arguments = getattr(func_obj, "arguments", "") if func_obj else ""
            tc = ToolCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
                type=call_type,
            )
            calls.append(tc)
        return calls

    @staticmethod
    def _parse_usage(usage_data: Any) -> UsageBreakdown | None:
        """
        Parse the usage field from the API response.

        Parameters
        ----------
        usage_data : Any
            The usage data from the API.

        Returns
        -------
        UsageBreakdown | None
            The parsed usage breakdown or `None` if not present.
        """
        if not usage_data:
            return None
        completion_details = getattr(
            usage_data, "completion_tokens_details", None
        )
        if not completion_details:
            return UsageBreakdown(
                prompt_tokens=usage_data.prompt_tokens,
                completion_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
            )
        return UsageBreakdown(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
            reasoning_tokens=completion_details.reasoning_tokens,
            audio_tokens=completion_details.audio_tokens,
            accepted_prediction_tokens=completion_details.accepted_prediction_tokens,
            rejected_prediction_tokens=completion_details.rejected_prediction_tokens,
        )
