from __future__ import annotations

from abc import ABC, abstractmethod

from aisemblies.responses import AssistantResponse
from aisemblies.tools import ToolCall
from aisemblies.utils import coerce_to_dict


class ContentPart(ABC):
    """
    Abstract base class for different content parts in a message.
    """

    @abstractmethod
    def to_msg(self) -> dict:
        """
        Convert the content part into a dictionary accepted by `/chat/completions`.

        Returns
        -------
        dict
            A dictionary representing this content part.
        """
        ...

    def fill_template(self, context: dict) -> ContentPart:
        """
        Return a new content part with placeholders replaced by context.

        Parameters
        ----------
        context : dict
            A dictionary containing values to replace placeholders in the content.

        Returns
        -------
        ContentPart
            A new instance of the content part with placeholders filled.
        """
        return self


class TextContent(ContentPart):
    """
    Represents plain text in a message.
    """

    def __init__(self, text: str):
        """
        Initialize TextContent with the given text.

        Parameters
        ----------
        text : str
            The plain text content.
        """
        self.text = text

    def to_msg(self) -> dict:
        """
        Convert the text content into a dictionary format.

        Returns
        -------
        dict
            A dictionary with type "text" and the text content.
        """
        return {"type": "text", "text": self.text}

    def fill_template(self, context: dict) -> TextContent:
        """
        Replace placeholders in the text with values from the context.

        Parameters
        ----------
        context : dict
            A dictionary containing values to replace placeholders in the text.

        Returns
        -------
        TextContent
            A new TextContent instance with placeholders filled.
        """
        return TextContent(self.text.format(**context))


class ImageContent(ContentPart):
    """
    Represents an image in a message.
    """

    def __init__(self, url: str, detail: str = "auto"):
        """
        Initialize ImageContent with the image URL and detail level.

        Parameters
        ----------
        url : str
            The URL of the image.
        detail : str, optional
            The level of detail for the image (default is "auto").
        """
        self.url = url
        self.detail = detail

    def to_msg(self) -> dict:
        """
        Convert the image content into a dictionary format.

        Returns
        -------
        dict
            A dictionary with type "image_url" and the image details.
        """
        return {
            "type": "image_url",
            "image_url": {"url": self.url, "detail": self.detail},
        }


class AudioContent(ContentPart):
    """
    Represents audio data in a message.
    """

    def __init__(self, data: str, fmt: str):
        """
        Initialize AudioContent with base64-encoded audio data and format.

        Parameters
        ----------
        data : str
            Base64-encoded audio data.
        fmt : str
            Format of the audio data (e.g., 'wav', 'mp3').
        """
        self.data = data
        self.format = fmt

    def to_msg(self) -> dict:
        """
        Convert the audio content into a dictionary format.

        Returns
        -------
        dict
            A dictionary with type "input_audio" and the audio details.
        """
        return {
            "type": "input_audio",
            "input_audio": {"data": self.data, "format": self.format},
        }


class ChatMessage(ABC):
    """
    Abstract base class for a chat message.
    """

    def __init__(self, role: str, content: str | list[ContentPart]):
        """
        Initialize a ChatMessage with a role and content.

        Parameters
        ----------
        role : str
            The role of the message sender (e.g., 'system', 'user', 'assistant').
        content : str | list[ContentPart]
            The content of the message, either as a plain string or a list of ContentPart objects.
        """
        self.role = role
        if isinstance(content, str):
            self._content_parts: list[ContentPart] | None = None
            self._single_string: str | None = content
        else:
            self._content_parts = content
            self._single_string = None

    def to_msg(self) -> dict:
        """
        Convert this message into a dictionary format accepted by `/chat/completions`.

        Returns
        -------
        dict
            A dictionary representing this message, containing its role and content.
        """
        if self._content_parts is None and self._single_string is not None:
            return {"role": self.role, "content": self._single_string}
        parts = [c.to_msg() for c in (self._content_parts or [])]
        return {"role": self.role, "content": parts}

    def render(self, context_obj) -> ChatMessage:
        """
        Create a new message with all placeholders in content filled using `context_obj`.

        Parameters
        ----------
        context_obj : Any
            An object containing values to replace placeholders in the content.

        Returns
        -------
        ChatMessage
            A new instance of ChatMessage with placeholders filled.
        """
        context = coerce_to_dict(context_obj)
        if self._single_string is not None:
            new_content = self._single_string.format(**context)
            return self.__class__(new_content)
        new_parts = [
            part.fill_template(context) for part in self._content_parts
        ]
        return self.__class__(new_parts)


class SystemMessage(ChatMessage):
    """
    A system prompt
    """

    def __init__(self, content: str | list[ContentPart]):
        """
        Initialize a SystemMessage with the given content.

        Parameters
        ----------
        content : str | list[ContentPart]
            The content of the system message.
        """
        super().__init__("system", content)


class UserMessage(ChatMessage):
    """
    A message from the end user.
    """

    def __init__(self, content: str | list[ContentPart]):
        """
        Initialize a UserMessage with the given content.

        Parameters
        ----------
        content : str | list[ContentPart]
            The content of the user message.
        """
        super().__init__("user", content)


class AssistantMessage(ChatMessage):
    """
    A message modeling the response from the model (assistant).
    """

    def __init__(
        self,
        content: str | list[ContentPart],
        refusal: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ):
        """
        Initialize an AssistantMessage with content, optional refusal, and tool calls.

        Parameters
        ----------
        content : str | list[ContentPart]
            The main content of the assistant's message.
        refusal : str, optional
            If present, indicates a refusal to comply, containing the refusal text.
        tool_calls : list[ToolCall], optional
            If present, any tool calls (function calls) generated by the model.
        """
        super().__init__("assistant", content)
        self.refusal = refusal
        self.tool_calls = tool_calls

    def to_msg(self) -> dict:
        """
        Convert to a dictionary in the style of an OpenAI assistant message.

        Returns
        -------
        dict
            Dictionary with role "assistant", content, refusal, and tool calls if any.
        """
        base_dict = super().to_msg()
        if self.refusal is not None:
            base_dict["refusal"] = self.refusal
        if self.tool_calls:
            base_dict["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": tc.type,
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return base_dict

    @classmethod
    def from_response(
        cls, response: AssistantResponse, choice_index: int = 0
    ) -> AssistantMessage:
        """
        Build an AssistantMessage from a given choice in an AssistantResponse.

        Parameters
        ----------
        response : AssistantResponse
            The response object from which to build the assistant message.
        choice_index : int, optional
            Which choice to extract from the response (default is 0).

        Returns
        -------
        AssistantMessage
            AssistantMessage built from the choice.
        """
        if not response.choices:
            return cls(content="")
        chosen = response.choices[choice_index]
        msg_data = chosen.message
        content = msg_data.content or ""
        return cls(
            content=content,
            refusal=msg_data.refusal,
            tool_calls=msg_data.tool_calls,
        )


class ToolMessage(ChatMessage):
    """
    A message modeling a tool (function) invocation.
    """

    def __init__(self, content: str | list[ContentPart], tool_call_id: str):
        """
        Initialize a ToolMessage with content and a tool call ID.

        Parameters
        ----------
        content : str | list[ContentPart]
            The content resulting from the tool invocation.
        tool_call_id : str
            The unique ID of the tool call.
        """
        super().__init__("tool", content)
        self.tool_call_id = tool_call_id

    def to_msg(self) -> dict:
        """
        Convert the tool message into a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the tool call ID and content.
        """
        base_dict = super().to_msg()
        base_dict["tool_call_id"] = self.tool_call_id
        return base_dict
