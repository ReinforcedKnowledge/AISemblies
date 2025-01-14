import json
from typing import Any

from aisemblies.messages import ToolMessage
from aisemblies.responses import AssistantResponse
from aisemblies.tools import ToolCollection


def invoke_llm_tool_calls(
    response: AssistantResponse,
    tool_collection: ToolCollection,
    raise_on_unknown_tool: bool = True,
    choice_idx: int | None = None,
) -> list[dict[str, Any]]:
    """
    Invoke the tool calls that the Large Language Model decided to make.

    Parameters
    ----------
    response : AssistantResponse
        The assistant response containing possible tool calls.
    tool_collection : ToolCollection
        A collection of available tools that can be invoked by the LLM.
    raise_on_unknown_tool : bool, default=True
        If `True`, raises a `ValueError` if the LLM requests a tool that's not found.
    choice_idx : int or None, optional
        If provided, only handle the tool calls for the choice at this index.
        If `None`, handle tool calls for all choices.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing:
        - "name": The name of the tool called
        - "call_id": The unique ID of the tool call
        - "output": The result of the tool function invocation
    """
    selected_choices = (
        [response.choices[choice_idx]]
        if choice_idx is not None
        else response.choices
    )
    results = []

    for choice in selected_choices:
        if not choice.message.tool_calls:
            continue

        for call in choice.message.tool_calls:
            tool_obj = tool_collection.get_tool_by_name(call.name)
            if tool_obj is None:
                error_msg = f"Unknown tool called by model: {call.name}"
                if raise_on_unknown_tool:
                    raise ValueError(error_msg)
                else:
                    print(error_msg)
                    continue

            try:
                call_args = json.loads(call.arguments)
            except json.JSONDecodeError as exc:
                error_msg = f"Invalid JSON in arguments for tool '{call.name}': {call.arguments}"
                if raise_on_unknown_tool:
                    raise ValueError(error_msg) from exc
                else:
                    print(error_msg)
                    continue

            output = tool_obj(**call_args)
            results.append(
                {
                    "name": call.name,
                    "call_id": call.call_id,
                    "output": output,
                }
            )

    return results


def generate_tool_messages_from_llm_calls(
    response: AssistantResponse,
    tool_collection: ToolCollection,
    raise_on_unknown_tool: bool = True,
    choice_idx: int | None = None,
) -> list[ToolMessage]:
    """
    Invoke the LLM-chosen tool calls and wrap their results in `ToolMessage` objects.

    Parameters
    ----------
    response : AssistantResponse
        The assistant response containing possible tool calls.
    tool_collection : ToolCollection
        A collection of available tools that can be invoked by the LLM.
    raise_on_unknown_tool : bool, default=True
        If `True`, raises a `ValueError` if the LLM requests a tool that's not found.
    choice_idx : int or None, optional
        If provided, only handle the tool calls for the choice at this index.
        If `None`, handle tool calls for all choices.

    Returns
    -------
    list[ToolMessage]
        A list of `ToolMessage` objects wrapping the outputs from all successful tool calls.
    """
    try:
        invocation_results = invoke_llm_tool_calls(
            response=response,
            tool_collection=tool_collection,
            raise_on_unknown_tool=raise_on_unknown_tool,
            choice_idx=choice_idx,
        )
    except Exception as exc:
        raise exc

    tool_messages = []
    for result in invocation_results:
        msg = ToolMessage(
            content=str(result["output"]), tool_call_id=result["call_id"]
        )
        tool_messages.append(msg)

    return tool_messages
