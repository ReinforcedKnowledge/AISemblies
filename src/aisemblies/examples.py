from openai import OpenAI

from aisemblies.messages import AssistantMessage, SystemMessage, UserMessage
from aisemblies.responses import AssistantResponse
from aisemblies.tool_helpers import generate_tool_messages_from_llm_calls
from aisemblies.tools import (
    FunctionTool,
    ToolCollection,
)


def fetch_stock_price(ticker: str) -> dict:
    """
    Fetch the current stock price for a given ticker symbol.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol, e.g. 'AAPL'.

    Returns
    -------
    dict
        The stock price information.
    """
    return {"ticker": ticker.upper(), "price": "150.00", "currency": "USD"}


def fetch_news_headlines(topic: str) -> dict:
    """
    Fetch the latest news headlines for a given topic.

    Parameters
    ----------
    topic : str
        A news topic, e.g. 'technology'.

    Returns
    -------
    dict
        A dictionary of headlines for the topic.
    """
    return {
        "topic": topic,
        "headlines": [
            f"Latest news in {topic}: Headline 1",
            f"Updates in {topic}: Headline 2",
            f"Insights in {topic}: Headline 3",
        ],
    }


def main():
    # Prepare tools
    stock_tool = FunctionTool(
        func=fetch_stock_price, description="Get the current stock price."
    )
    news_tool = FunctionTool(
        func=fetch_news_headlines, description="Get the latest news headlines."
    )
    tools = ToolCollection([stock_tool, news_tool])

    # Prepare initial messages
    system_msg = SystemMessage("You are a financial assistant.")
    user_msg = UserMessage(
        "What's the AAPL stock price and the latest technology news?"
    )

    # Convert to dict format
    messages_dict = [m.to_msg() for m in [system_msg, user_msg]]
    tools_dict = tools.to_openai_list()

    # Call the OpenAI API
    client = OpenAI()
    raw_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_dict,
        tools=tools_dict,
        tool_choice="auto",
    )

    # Parse the response
    parsed = AssistantResponse.from_completion(raw_response)
    print("First choice:", parsed.first_choice)

    # Invoke the suggested tool calls (if any) and convert outputs to messages
    tool_messages = generate_tool_messages_from_llm_calls(parsed, tools)
    for tm in tool_messages:
        print("Tool response:", tm.to_msg())

    # Continue the conversation with the tool outputs
    messages_dict.append(AssistantMessage.from_response(parsed).to_msg())
    messages_dict.extend([tm.to_msg() for tm in tool_messages])
    messages_dict.append(UserMessage("Thanks! Any final tips?").to_msg())

    # Next completion using updated conversation
    next_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_dict,
        tools=tools_dict,
        tool_choice="auto",
    )
    parsed_next = AssistantResponse.from_completion(next_response)
    print("Next response:", parsed_next.first_choice)


if __name__ == "__main__":
    main()
