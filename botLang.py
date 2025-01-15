import os
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
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# langraph modules to manage chatbot memory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# modules to modify the base message
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


# update State class to add language parameter
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str




load_dotenv()
groq_token = os.getenv('GROQ_API_KEY')

def load_character_card():
    card_path = os.getenv("CHARACTER_CARD_PATH")
    with open(card_path, 'r') as file:
        return file.read()

character_card = load_character_card()

client = Groq(
    api_key=groq_token,
)


model = ChatGroq(model="mixtral-8x7b-32768")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            character_card + """
            \n===LANGUAGE CONTROL===
            Current response language: {language}
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Define a new graph
workflow = StateGraph(state_schema=State)


# Define the function that calls the model
async def call_model(state: State):
    prompt = await prompt_template.ainvoke(state)
    response = await model.ainvoke(prompt)
    return {"messages": [response]}
    

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Wrap the execution in an async function
async def main():
    config = {"configurable": {"thread_id": "abc123"}}

    query = "Salut, ma petite chaudiere, moi c'est Armand, et toi?."
    language = "French"

    input_messages = [HumanMessage(query)]

    # Async invocation:
    output = await app.ainvoke({"messages": input_messages, "language": language}, config)
    output["messages"][-1].pretty_print()

    query = "Juste pour etre sure que tu suives, je m'appelle comment ma douce?"

    input_messages = [HumanMessage(query)]
    
    # Async invocation:
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()


# Run the async function
asyncio.run(main())




# system_template = character_card

# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", "{text}")]
# )

# prompt = prompt_template.invoke({"text": "Salut bb, ca va?"})


# response = model.invoke(prompt)
# print(response)

# def shadow_ai(user_message, message_history=[]):
#     messages = [
#         {"role": "system", "content": character_card}
#     ]
    
#     # Add conversation history
#     messages.extend(message_history)
    
#     # Add new user message
#     messages.append({"role": "user", "content": user_message})

#     response = client.chat.completions.create(
#         messages=messages,
#         model="mixtral-8x7b-32768",
#         temperature=0.5,
#         max_tokens=1024,
#         top_p=1,
#         stop=None,
#         stream=False,
#     )

#     # Add assistant's response to history
#     message_history.append({"role": "user", "content": user_message})
#     message_history.append({"role": "assistant", "content": response.choices[0].message.content})

#     return response


# # Wrap the Django ORM operations with sync_to_async
# get_user_data = sync_to_async(UserData.objects.get)
# save_user_data = sync_to_async(lambda x: x.save())

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
#         await update.message.reply_text(f'Welcome to ShadowMe!')


# async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
# 	user_message = update.message.text 
# 	await update.message.reply_text(shadow_ai(user_message).choices[0].message.content)


# def main():
#     try:
#         print("Starting bot initialization")  # Debug print
#         load_dotenv()
#         token = os.getenv('TOKEN')
#         print("Token loaded")  # Debug print

#         # # Create scheduler with explicit timezone
#         # scheduler = AsyncIOScheduler(timezone=pytz.UTC)

#         application = Application.builder().token(token).build()
#         print("Application built")  # Debug print
        
#         application.add_handler(CommandHandler('start', start))
#         print("Command Handler added")  # Debug print

#         # Handle bot response to message
#         application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo)) #handles text message
#         print("Message Handler added")  # Debug print

        
#         print("Starting polling")  # Debug print
#         application.run_polling()
#     except Exception as e:
#         print(f"Error in main: {str(e)}")
#         print(f"Error type: {type(e)}")
#         raise  # Re-raise the exception to see the full traceback


# if __name__ == '__main__':
#     main()