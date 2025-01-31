from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from aisemblies.messages import SystemMessage, UserMessage
from aisemblies.responses import (
    StreamedResponseBuilder,
)


async def stream_openai_response(
    completion, console, style: str = "bold cyan"
) -> str:
    """Aggregate streamed OpenAI chunks with partial Rich display."""
    aggregator = StreamedResponseBuilder()
    async for chunk in completion:
        aggregator.update_from_chunk(chunk)
        if chunk.choices:
            chunk_txt = chunk.choices[0].delta.content
        else:
            chunk_txt = ""
        console.print(chunk_txt, style=style, end="")

    console.print()
    final = aggregator.build_final_response()
    return final.first_choice.message.content


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()


async def retrieve(running_load: dict[str, Any]) -> str:
    print("\n[retrieve] Fetching documents from vector store ...")

    question = running_load["question"]
    if not question:
        print(
            "[retrieve] No question found in running_load, returning 'NO_QUESTION'."
        )
        return "NO_QUESTION"

    retrieved_docs = await retriever.ainvoke(question)

    running_load["documents"] = retrieved_docs
    print(f"[retrieve] Found {len(retrieved_docs)} documents.")

    return "OK"


async def grade_documents(running_load: dict[str, Any]) -> str:
    print("\n[grade_documents] Checking relevance of retrieved documents ...")
    question = running_load["question"]
    docs = running_load["documents"]
    if not docs:
        print(
            "[grade_documents] No documents to grade. Returning 'IRRELEVANT'."
        )
        running_load["search_decision"] = "Yes"
        return "IRRELEVANT"

    system_prompt = SystemMessage("""
You are a grader assessing relevance of a retrieved document to a user question.\n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n
Give a binary 'yes' or 'no' score to indicate whether the document is relevant to the question.
""")
    system_message = system_prompt.to_msg()

    user_prompt = UserMessage("""
Here is the retrieved document: \n\n {context} \n\n
Here is the user question: {question} \n
""")

    relevant_docs = []
    for doc in docs:
        completion = await running_load["client"].chat.completions.create(
            model="gpt-4o",
            messages=[
                system_message,
                user_prompt.render(
                    {"context": doc, "question": question}
                ).to_msg(),
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        grade = await stream_openai_response(
            completion, running_load["console"], style="bold green"
        )
        if "yes" in grade.lower():
            relevant_docs.append(doc)

    if len(relevant_docs) > 0:
        print(
            f"[grade_documents] Found {len(relevant_docs)} relevant document(s)."
        )
        running_load["documents"] = relevant_docs
        running_load["search_decision"] = "No"
        return "RELEVANT"
    else:
        print("[grade_documents] All docs are irrelevant => need web search.")
        running_load["search_decision"] = "Yes"
        running_load["documents"] = []
        return "IRRELEVANT"


async def transform_query(running_load: dict[str, Any]) -> str:
    print(
        "\n[transform_query] Attempting to rewrite the query for better web search ..."
    )

    system_prompt = SystemMessage("""
You are generating questions that are well optimized for retrieval.\n
Look at the input and try to reason about the underlying semantic intent/meaning.\n
""")
    system_message = system_prompt.to_msg()

    user_prompt = UserMessage("""
Here is the initial question:
\n ------- \n
{question}
\n ------- \n
Formulate an improved question:
""")
    question = running_load["question"] or ""
    user_message = user_prompt.render({"question": question}).to_msg()

    completion = await running_load["client"].chat.completions.create(
        model="gpt-4o",
        messages=[
            system_message,
            user_message,
        ],
        stream=True,
        stream_options={"include_usage": True},
    )
    better_question = await stream_openai_response(
        completion, running_load["console"], style="bold magenta"
    )

    running_load["question"] = better_question
    print(f"[transform_query] Transformed question => {better_question}")

    return "TRANSFORMED"


async def web_search(running_load: dict[str, Any]) -> str:
    print("\n[web_search] Searching with Tavily ...")
    question = running_load["question"]
    tool = TavilySearchResults()
    results = await tool.ainvoke({"query": question})
    new_docs = [Document(page_content=r["content"]) for r in results]

    running_load["documents"].extend(new_docs)
    print(f"[web_search] Received {len(new_docs)} additional web results.")

    return "DONE"


async def generate(running_load: dict[str, Any]) -> str:
    print(
        "\n[generate] Generating an answer using the retrieved documents + question ..."
    )

    question = running_load["question"] or "No question found"
    docs = running_load["documents"]

    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = SystemMessage("""
You are an assistant for question-answering tasks. Use the pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
""")
    system_message = system_prompt.to_msg()

    user_prompt = UserMessage("""
Question: {question}
Context: {context}
Answer:
""")
    user_message = user_prompt.render(
        {"question": question, "context": context}
    ).to_msg()

    completion = await running_load["client"].chat.completions.create(
        model="gpt-4o",
        messages=[
            system_message,
            user_message,
        ],
        stream=True,
        stream_options={"include_usage": True},
    )
    result = await stream_openai_response(
        completion, running_load["console"], style="bold cyan"
    )

    running_load["generation"] = result
    print(f"[generate] Final answer:\n{result}")

    return "DONE"


async def error_handler(
    running_load: dict[str, Any], exception, traceback_str
) -> None:
    """
    If any of our 'C-RAG' steps raises an exception, we will jump here.
    """
    print("[error_handler] Shutting down.")
    return None
