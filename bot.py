import os
import sys
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shadow_bot.settings')
django.setup()


from shadow_bot.settings import DATABASES
from user_data.models import UserData
from telegram import Update
from asyncio import Queue
from telegram.ext import Updater, CommandHandler, CallbackContext
from dotenv import load_dotenv


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


def main() -> None:
	load_dotenv()
	token = os.getenv('TOKEN')
	# botQueue = Queue()
	updater = Updater(token)
	dispatcher = updater.dispatcher
	dispatcher.add_handler(CommandHandler('start', start))
	updater.start_polling()
	updater.idle()


if __name__ == '__main__':
	main()