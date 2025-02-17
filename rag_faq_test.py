import getpass
import os
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_mistralai import ChatMistralAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, MessagesState, StateGraph
from typing_extensions import List, TypedDict

# modules to implement retrieval in conversation chatbot
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
# memory
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate



load_dotenv()
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = 
os.getenv('LANGSMITH_TRACING')
os.getenv('LANGSMITH_API_KEY')

# set up model
os.getenv('MISTRAL_API_KEY')

# llm = ChatMistralAI(model="open-mixtral-8x22b")
llm = ChatMistralAI(model="mistral-large-latest")


# RAG Set Up
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# in memory set up / need other set up when connect to DB
vector_store = InMemoryVectorStore(embeddings)


# Get the hidden sources
def load_file(file_name):
    card_path = os.getenv(file_name)
    with open(card_path, 'r') as file:
        return file.read()


rag_source = load_file("FAQ_PATH") # rag source FAQ
character_card = load_file("CHARACTER_CARD_PATH") # system prompt

docs = [
    Document(
        page_content=rag_source,
        metadata={"source": "FAQ", "title": "Frequently Asked Questions"}
    )
]

# print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)


# Index chunks
_ = vector_store.add_documents(documents=all_splits)

graph_builder = StateGraph(MessagesState)


# Create a tool for the retrieval, allows better query to search / or respond directly
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    conversation = [SystemMessage(character_card)] + state["messages"]

    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(conversation)

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = f""" 
        Tu peux t'aider du document ci-dessous si la réponse ne t'es pas évidente. 
        Si tu ne trouves pas la réponse là-dedans, réponds simplement que tu ne sais pas.

        Document:
        {docs_content}
            """
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages # try to append to system m,essage

    response = llm.invoke(prompt)
  
    return {"messages": [response]}


# Compile application and test

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)


# implement memory
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}


input_message = "As tu des videos gratuites à me donner?"

for step in graph.stream(
    {"messages": [
        {
            "role": "human", 
            "content": input_message
        },
    ]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()
