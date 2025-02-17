import os
import getpass
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shadow_bot.settings')
django.setup()

import asyncio
from groq import Groq
from shadow_bot.settings import DATABASES
from user_data.models import UserData
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from asgiref.sync import sync_to_async

# langchain
from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage, 
    HumanMessage, 
    BaseMessage, # Added to modify our State class
    AIMessage,  
    trim_messages # Manage Messages History
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# langraph modules to manage chatbot memory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# modules to modify the base message
from typing import Sequence
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# modules for RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# Configure the chatbot using LangChain, work in progess
# update State class to add language parameter
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# # Enabling LangSmith
# os.environ["LANGSMITH_TRACING"] = "false"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
# Disable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)  # Remove API key if present

load_dotenv()
# get the Groq API Key
groq_token = os.getenv('GROQ_API_KEY')

client = Groq(
    api_key=groq_token,
)


# RAG Set Up
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)


# Get the hidden sources
def load_file(file_name):
    card_path = os.getenv(file_name)
    with open(card_path, 'r') as file:
        return file.read()


rag_source = load_file("FAQ_PATH") # rag source FAQ
character_card = load_file("CHARACTER_CARD_PATH") # system prompt


# Initiate the Chatbot config
model = ChatGroq(model="mixtral-8x7b-32768")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            character_card
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Define a new graph
workflow = StateGraph(state_schema=State)

# Add trimmer
trimmer = trim_messages(
    max_tokens=6000,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Initialize Messages History
messages = []


# Define the function that calls the model
async def call_model(state: State):
    trimmed_messages = await trimmer.ainvoke(state["messages"])
    prompt = await prompt_template.ainvoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = await model.ainvoke(prompt)
    return {"messages": [response]}
    

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# Wrap the execution in an async function to handle multiple users
async def shadow_ai(user_message):
    config = {"configurable": {"thread_id": "abc123"}}

    query = user_message
    language = "Francais"

    input_messages = messages + [HumanMessage(query)]

    # Async invocation:
    output = await app.ainvoke({"messages": input_messages, "language": language}, config)

    return output["messages"][-1].content

# # Run the async function
# asyncio.run(main())


# Wrap the Django ORM operations with sync_to_async
get_user_data = sync_to_async(UserData.objects.get)
save_user_data = sync_to_async(lambda x: x.save())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Start command received")  # Debug print
    user_id = update.effective_user.id 
    try:
        user_data = await get_user_data(id=user_id)
        await update.message.reply_text(f'Welcome back {user_data.first_name}!')
    except UserData.DoesNotExist:
        new_user = UserData(
            id=user_id,
            first_name=update.effective_user.first_name,
            last_name=update.effective_user.last_name,
            username=update.effective_user.username
        )
        await save_user_data(new_user)
        await update.message.reply_text(f'Welcome to ShadowMe!')


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_message = update.message.text 
        response = await shadow_ai(user_message)  # awaited here
        if response:  # make sure we have a response
            await update.message.reply_text(str(response))  # convert to string explicitly
        else:
            await update.message.reply_text("No response generated")
    except Exception as e:
        print(f"Error in echo handler: {str(e)}")
        await update.message.reply_text("Sorry, I encountered an error processing your message.")


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

        # Handle bot response to message
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo)) #handles text message
        print("Message Handler added")  # Debug print

        
        print("Starting polling")  # Debug print
        application.run_polling()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print(f"Error type: {type(e)}")
        raise  # Re-raise the exception to see the full traceback


if __name__ == '__main__':
    main()

