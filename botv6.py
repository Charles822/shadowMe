import os
import getpass
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shadow_bot.settings.dev')
django.setup()

import asyncio
import random
from django.conf import settings

# Access DATABASES via the Django settings object
DATABASES = settings.DATABASES
print(settings.DATABASES)

from user_data.models import UserData
from telegram import Update, ChatFullInfo, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction
from dotenv import load_dotenv
from asgiref.sync import sync_to_async


from IPython.display import Image, display
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere

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
llm = ChatMistralAI(model="mistral-large-latest")
# llm = ChatCohere(model="command-r-plus")


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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
all_splits = text_splitter.split_documents(docs)


# Index chunks
_ = vector_store.add_documents(documents=all_splits)

graph_builder = StateGraph(MessagesState)


# conversation = [SystemMessage(character_card)]

# TOOLS 
# Create a tool for the retrieval, allows better query to search / or respond directly
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    print('RETRIEVE TOOL HAS BEEN CALLED')
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs

@tool
async def alert_creator_tool() -> str:
    """ Call the alert_creator function when the user is shows sign of being ready to pay."""
    await alert_creator(1002947084, 'sofiatilla')
    print('ALERT CREATOR TOOL HAS BEEN CALLED')
    return "Message to creator has been sent."

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    conversation = [SystemMessage(character_card)] + state["messages"]

    llm_with_tools = llm.bind_tools([alert_creator_tool, retrieve])
    response = llm_with_tools.invoke(conversation)

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools = ToolNode([alert_creator_tool, retrieve])


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
    # system_message_content = f"""
    #     Ton role principal: 
    #     {character_card}
    #     Tu peux t'aider du document ci-dessous si la réponse ne t'es pas évidente. 
    #     Si tu ne trouves pas la réponse là-dedans, réponds simplement que tu ne sais pas.

    #     Document:
    #     {docs_content}
    #         """
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(character_card)] + conversation_messages # try to append to system message

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

async def shadow_ai(user_message, thread_id):
    config = {"configurable": {"thread_id": thread_id}}

    query = user_message
    # language = "Francais"

    # input_messages = messages + [HumanMessage(query)]

    # Async invocation:
    output = await graph.ainvoke({"messages": [HumanMessage(query)]}, config)

    return output["messages"][-1].content


# telegram bot part

# bot_token = os.environ['TOKEN']
# bot = Bot(token=bot_token)

# Wrap the Django ORM operations with sync_to_async
get_user_data = sync_to_async(UserData.objects.get)
save_user_data = sync_to_async(lambda x: x.save())

# (creator_business_id, client_business_id) -> bool
# human_takeover_flags = {
#     ('Sirebrown', 'sofiatilla'): False,
# }

# initiate human flag
human_takeover_flags = {}

# user_chat_id = 1002947084

async def alert_creator(user_chat_id, client_username):
    print('Debug Creator ID in Alert Creator', user_chat_id, client_username)
    print('DEBUG ALERT CREATOR TRIGGERED')

    try:
        # response = await bot.send_message(
        #     chat_id=user_chat_id,
        #     text=f"Client {client_username} signaled readiness to pay!"
        # )
        # print("Message sent successfully:", response)
        print('TOOL SUCCESSFULLY CALLED')
    except Exception as e:
        print("Error sending message:", e)


# async def human_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     # 1. Identify the creator
#     creator_username = update.effective_user.username
#     print('Creator username accessed via Human On', creator_username)

#     # 2. Parse the target client ID from args
#     command_args = context.args
#     if not command_args:
#         await update.message.reply_text("Usage: /human_on <client_username>")
#         return
#     client_username = command_args[0]
#     print(client_username)

#     # 3. Set the flag
#     human_takeover_flags[(creator_username, client_username)] = True

#     # 4. Confirm
#     await update.message.reply_text(
#         f"Human takeover ON for client `{client_username}` under creator `{creator_username}`."
#     )

# async def human_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     # 1. Identify the creator
#     creator_username = update.effective_user.username
#     print('Creator username accessed via Human Off', creator_username)

#     # 2. Parse the target client ID from args
#     command_args = context.args
#     if not command_args:
#         await update.message.reply_text("Usage: /human_off <client_business_id>")
#         return
#     client_username = command_args[0]
#     print(client_username)

#     # 3. Toggle off
#     human_takeover_flags[(creator_username, client_username)] = False

#     # 4. Confirm
#     await update.message.reply_text(
#         f"Human takeover OFF for client `{client_username}` under creator `{creator_username}`."
#     )

def start(update: Update, context: CallbackContext) -> None:
	user_id = update.effective_user.id 
	try:
		user_data = UserData.objects.get(id=user_id)
		update.message.reply_text(f'Welcome back {user_data.first_name}!')
	except UserData.DoesNotExist:
		new_user = UserData(
			id = user_id,
			first_name = update.effective_user.first_name,
			last_name = update.effective_user.last_name,
			username = update.effective_user.username
		)
		new_user.save()
		update.message.reply_text(f'Welcome to ShadowMe!')


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # print('inspect chat triggered')
    # chat_id = update.effective_chat.id
    # # or you might use update.effective_message.chat_id
    # # print(chat_id)

    # chat_info = await context.bot.get_chat(chat_id)
    # # chat_info can be an instance of either Chat or ChatFullInfo
    # # print(chat_info)

    # # For a business chat, you may have the extended attributes:
    # if isinstance(chat_info, ChatFullInfo):
    #     # Potentially available fields
    #     print("full title:", chat_info.title)
    #     print("business_intro:", chat_info.business_intro)
    #     # etc.
    # else:
    #     # It's just a basic Chat object
    #     print("This is a normal Chat object, not ChatFullInfo")
    
    # print('BusinessConnection triggered')

    # print(update.to_dict())

    # bc = update.business_message.business_connection_id

    # business_connection = chat_info = await context.bot.get_business_connection(bc)

    # print(business_connection)


    telegram_id = update.effective_user.id 
    print(telegram_id)
    print(update.to_dict())


    try:
        print('Checking user data')
        user_data = await get_user_data(telegram_id=telegram_id)
        print('User already exist')
        # await update.business_message.reply_text(f'Welcome back {user_data.first_name}!')
    except UserData.DoesNotExist:
        new_user = UserData(
            telegram_id=telegram_id,
            first_name=update.effective_user.first_name,
            last_name=update.effective_user.last_name,
            username=update.effective_user.username
        )
        await save_user_data(new_user)
        # await update.business_message.reply_text(f'Welcome to ShadowMe!')
    print('The user', update.effective_user.first_name)

    try:
        user_message = update.business_message.text
        # Get creator ID
        print('Get creator ID')
        bc = update.business_message.business_connection_id
        business_connection = await context.bot.get_business_connection(bc)
        print(business_connection)
        creator_id = business_connection.user.username
        print('Debug Creator ID in Echo', creator_id)
        client_username = update.effective_user.username
        # test user data
        thread_id = creator_id + client_username 
        print(thread_id)

        if (creator_id, client_username) not in human_takeover_flags.keys():
            human_takeover_flags[(creator_id, client_username)] = False

        print('Setting human_takeover in echo, human_takeover equal:', human_takeover_flags[(creator_id, client_username)])

        response = await shadow_ai(user_message, thread_id)  # awaited here
        if response and human_takeover_flags[(creator_id, client_username)] == False:  # make sure we have a response
            # 1. Indicate the bot is "typing" for a short random duration
            print("DEBUG chat_id:", update.effective_chat.id)

            await context.bot.send_chat_action(chat_id=update.effective_chat.id, business_connection_id=bc, action=ChatAction.TYPING)

            # # 2. Wait a random time to simulate typing
            # # For example, between 1 and 3 seconds
            await asyncio.sleep(random.uniform(5, 10))

            # 3. Respond
            await update.business_message.reply_text(str(response))  # convert to string explicitly
        # else:
            # await update.business_message.reply_text("No response generated")
    except Exception as e:
        print(f"Error in echo handler: {str(e)}")
        await update.business_message.reply_text("Attend désolé je dois partir, je reviens tout à l'heure.")


def main():
    try:
        print("Starting bot initialization")  # Debug print
        load_dotenv()
        token = os.getenv('TOKEN')
        print("Token loaded")  # Debug print

        application = Application.builder().token(token).build()
        print("Application built")  # Debug print
        
        application.add_handler(CommandHandler('start', start))
        print("Command Handler added")  # Debug print

        # application.add_handler(CommandHandler('human_on', human_on))
        # print("Command human_on Handler added")  # Debug print

        # application.add_handler(CommandHandler('human_off', human_off))
        # print("Command human_on Handler added")  # Debug print

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

