import os
from telethon import TelegramClient, events
from dotenv import load_dotenv

load_dotenv()

# Telegram Env Variables
API_ID = os.environ['API_ID']
# Telegram Auth API HASH
API_HASH = os.environ['API_HASH']
# Telegram Bot API TOKEN generated from @botfather
BOT_TOKEN = os.environ['TOKEN']


client = TelegramClient('Sirebrown', api_id, api_hash)

@client.on(events.NewMessage)
async def my_event_handler(event):
    if 'hello' in event.raw_text:
        await event.reply('hi!')

client.start()
client.run_until_disconnected()