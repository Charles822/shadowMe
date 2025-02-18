import os
import getpass
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shadow_bot.settings')
django.setup()

import asyncio
from shadow_bot.settings import DATABASES
from user_data.models import UserData
from telegram import Update, ChatFullInfo
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from asgiref.sync import sync_to_async


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
from langchain_core.messages import (
    SystemMessage, 
    HumanMessage, 
    BaseMessage, # Added to modify our State class
    AIMessage,  
    trim_messages # Manage Messages History
)
from langgraph.prebuilt import ToolNode, tools_condition
# memory
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model


load_dotenv()

# Telegram Env Variables
API_ID = os.environ['API_ID']
# Telegram Auth API HASH
API_HASH = os.environ['API_HASH']


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = 
os.getenv('LANGSMITH_TRACING')
os.getenv('LANGSMITH_API_KEY')

# set up model
os.getenv('MISTRAL_API_KEY')
os.getenv('COHERE_API_KEY')

# llm = ChatMistralAI(model="open-mixtral-8x22b")
# llm = ChatMistralAI(model="mistral-large-latest")


llm = init_chat_model("command-r-plus", model_provider="cohere")

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


# input_message = "As tu des videos gratuites à me donner?"

async def shadow_ai(user_message):
    config = {"configurable": {"thread_id": "abc123"}}

    query = user_message
    # language = "Francais"

    # input_messages = messages + [HumanMessage(query)]

    # Async invocation:
    output = await graph.ainvoke({"messages": [HumanMessage(query)]}, config)

    return output["messages"][-1].content


# telegram bot part

# Wrap the Django ORM operations with sync_to_async
get_user_data = sync_to_async(UserData.objects.get)
save_user_data = sync_to_async(lambda x: x.save())


# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print("Start command received")  # Debug print
#     user_id = update.effective_user.id 
#     try:
#         user_data = await get_user_data(id=user_id)
#         await update.message.reply_text(f'Welcome back {user_data.first_name}!')
#     except UserData.DoesNotExist:
#         new_user = UserData(
#             id=user_id,
#             first_name=update.effective_user.first_name,
#             last_name=update.effective_user.last_name,
#             username=update.effective_user.username
#         )
#         await save_user_data(new_user)
#         await update.business_message.reply_text(f'Welcome to ShadowMe!')
#     print(update.effective_user.first_name)


# async def inspect_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print('inspect chat triggered')
#     chat_id = update.effective_chat.id
#     # or you might use update.effective_message.chat_id
#     print(chat_id)

#     chat_info = await context.bot.get_chat(chat_id)
#     # chat_info can be an instance of either Chat or ChatFullInfo
#     print(chat_info)

#     # For a business chat, you may have the extended attributes:
#     if isinstance(chat_info, ChatFullInfo):
#         # Potentially available fields
#         print("full title:", chat_info.title)
#         print("business_intro:", chat_info.business_intro)
#         # etc.
#     else:
#         # It's just a basic Chat object
#         print("This is a normal Chat object, not ChatFullInfo")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print('inspect chat triggered')
    chat_id = update.effective_chat.id
    # or you might use update.effective_message.chat_id
    # print(chat_id)

    chat_info = await context.bot.get_chat(chat_id)
    # chat_info can be an instance of either Chat or ChatFullInfo
    # print(chat_info)

    # For a business chat, you may have the extended attributes:
    if isinstance(chat_info, ChatFullInfo):
        # Potentially available fields
        print("full title:", chat_info.title)
        print("business_intro:", chat_info.business_intro)
        # etc.
    else:
        # It's just a basic Chat object
        print("This is a normal Chat object, not ChatFullInfo")
    
    print('BusinessConnection triggered')

    # print(update.to_dict())

    bc = update.business_message.business_connection_id

    business_connection = chat_info = await context.bot.get_business_connection(bc)

    print(business_connection)


    user_id = update.effective_user.id 

    try:
        user_data = await get_user_data(id=user_id)
        # await update.business_message.reply_text(f'Welcome back {user_data.first_name}!')
    except UserData.DoesNotExist:
        new_user = UserData(
            id=user_id,
            first_name=update.effective_user.first_name,
            last_name=update.effective_user.last_name,
            username=update.effective_user.username
        )
        await save_user_data(new_user)
        # await update.business_message.reply_text(f'Welcome to ShadowMe!')
    print('The user', update.effective_user.first_name)

    try:
        user_message = update.business_message.text
        # test user data

        response = await shadow_ai(user_message)  # awaited here
        if response:  # make sure we have a response
            await update.business_message.reply_text(str(response))  # convert to string explicitly
        else:
            await update.business_message.reply_text("No response generated")
    except Exception as e:
        print(f"Error in echo handler: {str(e)}")
        await update.business_message.reply_text("Sorry, I encountered an error processing your message.")


def main():
    try:
        print("Starting bot initialization")  # Debug print
        load_dotenv()
        token = os.getenv('TOKEN')
        print("Token loaded")  # Debug print

        application = Application.builder().token(token).build()
        print("Application built")  # Debug print
        
        # application.add_handler(CommandHandler('start', start))
        # print("Command Handler added")  # Debug print

        # Handle bot response to message
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo)) #handles text message
        print("Echo Message Handler added")  # Debug print

        # application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, inspect_chat)) #handles text message
        # print("Inspect Chat Message Handler added")  # Debug print
        
        print("Starting polling")  # Debug print
        application.run_polling()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print(f"Error type: {type(e)}")
        raise  # Re-raise the exception to see the full traceback


if __name__ == '__main__':
    main()

