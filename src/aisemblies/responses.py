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


def parse_usage(usage_data: Any) -> UsageBreakdown | None:
    """
    Parse the usage field from the completion.

    Parameters
    ----------
    usage_data : Any
        The usage data from `/chat/completions`.

    Returns
    -------
    UsageBreakdown | None
        The parsed usage breakdown or `None` if not present.
    """
    if not usage_data:
        return None

    completion_details = getattr(usage_data, "completion_tokens_details", None)
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


def parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall] | None:
    """
    Convert the `tool_calls` field into a list of `ToolCall`.

    Parameters
    ----------
    raw_tool_calls : Any
        Tools calls from `/chat/completions`.

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
        calls.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
                type=call_type,
            )
        )
    return calls


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
        list[AssistantChoice]
            All AssistantChoice instances in the response.
        """
        return self.choices

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        """
        Get all tool calls from all choices in the response.

        Returns
        -------
        list[ToolCall]
            All ToolCall instances found across choices.
        """
        calls: list[ToolCall] = []
        for choice in self.choices:
            if choice.message.tool_calls:
                calls.extend(choice.message.tool_calls)
        return calls

    @classmethod
    def from_blocking_completion(cls, data: Any) -> AssistantResponse:
        """
        Create an `AssistantResponse` from a `/chat/completions` response.
        To be used with blocking request.

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
        usage = parse_usage(usage_data)

        raw_choices = getattr(data, "choices", [])
        parsed_choices: list[AssistantChoice] = []

        for choice in raw_choices:
            finish_reason = getattr(choice, "finish_reason", None)
            index = getattr(choice, "index", 0)
            msg_data = getattr(choice, "message", None)

            if not msg_data:
                assistant_msg = AssistantResponseMessage()
            else:
                content = getattr(msg_data, "content", None)
                refusal = getattr(msg_data, "refusal", None)
                tool_calls_data = getattr(msg_data, "tool_calls", None)
                tool_calls = parse_tool_calls(tool_calls_data)
                assistant_msg = AssistantResponseMessage(
                    content=content,
                    refusal=refusal,
                    tool_calls=tool_calls,
                )

            parsed_choices.append(
                AssistantChoice(
                    message=assistant_msg,
                    finish_reason=finish_reason,
                    index=index,
                )
            )

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


class StreamedResponseBuilder:
    """
    Build a complete AssistantResponse from completion chunks.
    """

    def __init__(self):
        """
        Initializes the StreamedResponseBuilder with empty fields for in-progress data.
        """
        self.id = ""
        self.model = ""
        self.created = 0
        self.choices = []
        self.usage = None
        self.system_fingerprint = None
        self.service_tier = None
        self.object_type = None
        self._partial_tool_calls: dict[str, ToolCall] = {}

    def update_from_chunk(self, chunk: Any) -> None:
        """
        Merges partial data from a streaming chunk.

        Parameters
        ----------
        chunk : Any
            A streaming chunk object.
        """
        obj_type = getattr(chunk, "object", None)
        if obj_type:
            self.object_type = obj_type

        created_val = getattr(chunk, "created", None)
        if created_val is not None:
            self.created = created_val

        model_val = getattr(chunk, "model", None)
        if model_val is not None:
            self.model = model_val

        system_fp = getattr(chunk, "system_fingerprint", None)
        if system_fp is not None:
            self.system_fingerprint = system_fp

        service_tier = getattr(chunk, "service_tier", None)
        if service_tier is not None:
            self.service_tier = service_tier

        usage_val = getattr(chunk, "usage", None)
        if usage_val:
            usage_parsed = parse_usage(usage_val)
            if usage_parsed:
                self.usage = usage_parsed

        raw_choices = getattr(chunk, "choices", [])
        for raw_choice in raw_choices:
            finish_reason = getattr(raw_choice, "finish_reason", None)
            index = getattr(raw_choice, "index", 0)
            delta = getattr(raw_choice, "delta", None)

            while len(self.choices) <= index:
                empty_msg = AssistantResponseMessage()
                empty_choice = AssistantChoice(
                    message=empty_msg,
                    finish_reason=None,
                    index=len(self.choices),
                )
                self.choices.append(empty_choice)

            existing_choice = self.choices[index]
            if finish_reason is not None:
                existing_choice.finish_reason = finish_reason

            if not delta:
                continue

            partial_content = getattr(delta, "content", None)
            partial_refusal = getattr(delta, "refusal", None)
            tool_calls_data = getattr(delta, "tool_calls", None)

            msg = existing_choice.message
            if partial_content:
                msg.content = (msg.content or "") + partial_content

            if partial_refusal:
                msg.refusal = (msg.refusal or "") + partial_refusal

            if tool_calls_data:
                self._merge_partial_tool_calls(msg, tool_calls_data)

    def _merge_partial_tool_calls(
        self, msg: AssistantResponseMessage, tool_calls_data: Any
    ) -> None:
        """
        Accumulates partial tool-calling data from chunk into stored calls.

        Parameters
        ----------
        msg : AssistantResponseMessage
            The current response message.
        tool_calls_data : Any
            Tools data from the chunk.
        """
        if msg.tool_calls is None:
            msg.tool_calls = []

        for call_chunk in tool_calls_data:
            call_id = getattr(call_chunk, "id", None) or ""
            call_type = getattr(call_chunk, "type", None)
            func_obj = getattr(call_chunk, "function", None)

            if not call_id:
                call_id = "partial_noid"

            if call_id not in self._partial_tool_calls:
                self._partial_tool_calls[call_id] = ToolCall(
                    call_id=call_id, name="", arguments=""
                )

            partial_call = self._partial_tool_calls[call_id]

            if call_type:
                partial_call.type = call_type

            if func_obj:
                fn_name = getattr(func_obj, "name", None)
                fn_args = getattr(func_obj, "arguments", None)
                if fn_name:
                    partial_call.name = fn_name
                if fn_args:
                    partial_call.arguments += fn_args

        msg.tool_calls.clear()
        for tc in self._partial_tool_calls.values():
            msg.tool_calls.append(tc)

    def build_final_response(self) -> AssistantResponse:
        """
        Produces a final AssistantResponse after all streaming chunks.

        Returns
        -------
        AssistantResponse
            Fully aggregated response.
        """
        return AssistantResponse(
            id=self.id,
            model=self.model,
            created=self.created,
            choices=self.choices,
            usage=self.usage,
            system_fingerprint=self.system_fingerprint,
            service_tier=self.service_tier,
            object_type=self.object_type,
        )
