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


# Configure the chatbot using LangChain, work in progess
# update State class to add language parameter
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# Enabling LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

load_dotenv()
# get the Groq API Key
groq_token = os.getenv('GROQ_API_KEY')

client = Groq(
    api_key=groq_token,
)


# Get the hidden system prompt
def load_character_card():
    card_path = os.getenv("CHARACTER_CARD_PATH")
    with open(card_path, 'r') as file:
        return file.read()

character_card = load_character_card()


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

# #Add a series of messages history for testing purpose
messages = [
    # SystemMessage(content=character_card),
    HumanMessage(content=""),
    AIMessage(content=""),
    HumanMessage(content=""),
    AIMessage(content=""),
]

# trimmer.invoke(messages)


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
async def main():
    config = {"configurable": {"thread_id": "abc123"}}

    query = ""
    language = "Francais"

    input_messages = messages + [HumanMessage(query)]

    # Async invocation:
    output = await app.ainvoke({"messages": input_messages, "language": language}, config)
    output["messages"][-1].pretty_print()

    # query = "Juste pour etre sure que tu suives, quel est mon nom?"

    # input_messages = [HumanMessage(query)]
    
    # # Async invocation:
    # output = await app.ainvoke({"messages": input_messages}, config)
    # output["messages"][-1].pretty_print()


# Run the async function
asyncio.run(main())