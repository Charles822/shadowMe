import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shadow_bot.settings')
django.setup()

from groq import Groq
from shadow_bot.settings import DATABASES
from user_data.models import UserData
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from asgiref.sync import sync_to_async


# Bot set up using Telegram latest version
# Access environment variables
load_dotenv()

# Get the hidden system prompt
def load_character_card():
    card_path = os.getenv("CHARACTER_CARD_PATH")
    with open(card_path, 'r') as file:
        return file.read()

character_card = load_character_card()

# Get the groq token
groq_token = os.getenv('GROQ_API_KEY')

client = Groq(
    api_key=groq_token,
)


# Define the chatbot config using Groq
def shadow_ai(user_message, message_history=[]):
    messages = [
        {"role": "system", "content": character_card}
    ]
    
    # Add conversation history
    messages.extend(message_history)
    
    # Add new user message
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        messages=messages,
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    # Add assistant's response to history
    message_history.append({"role": "user", "content": user_message})
    message_history.append({"role": "assistant", "content": response.choices[0].message.content})

    return response


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
	user_message = update.message.text 
	await update.message.reply_text(shadow_ai(user_message).choices[0].message.content)


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