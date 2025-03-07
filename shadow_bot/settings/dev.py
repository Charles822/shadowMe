from .common import *

DEBUG = True

# SUPABASE SET UP

# Load environment variables from .env
load_dotenv()

# Fetch variables
SUPA_USER = os.getenv("SUPA_USER")
SUPA_PASSWORD = os.getenv("SUPA_PASSWORD")
SUPA_HOST = os.getenv("SUPA_HOST")
SUPA_PORT = os.getenv("SUPA_PORT")
SUPA_DBNAME = os.getenv("SUPA_DBNAME")

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': SUPA_DBNAME,
        'USER': SUPA_USER,
        'PASSWORD': SUPA_PASSWORD,
        'HOST': SUPA_HOST,
        'PORT': SUPA_PORT
    }
}
