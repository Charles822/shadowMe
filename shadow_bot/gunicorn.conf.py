import multiprocessing
import subprocess
import os

# # Gunicorn configuration
# workers = 2
# worker_class = 'gevent'  # or 'sync' if async is not needed
# accesslog = '-'
# errorlog = '-'
# loglevel = 'debug'

# Custom setup to run botv6.py
# def on_starting(server):
#     """Run the bot script when Gunicorn starts."""
#     # Set up Django environment for the bot script
#     os.environ.setdefault("DJANGO_SETTINGS_MODULE", "shadow_bot.settings.dev")

#     # Start the bot script as a subprocess
#     bot_process = subprocess.Popen(["python", "botv6.py"])
#     server.log.info(f"Started botv6.py with PID {bot_process.pid}")