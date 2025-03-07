from .common import *

DEBUG = False

ALLOWED_HOSTS = []

# SUPABASE SET UP

# Load environment variables from .env
load_dotenv()

# Fetch variables
SUPA_USER = os.getenv("SUPA_USER_PROD")
SUPA_PASSWORD = os.getenv("SUPA_PASSWORD_PROD")
SUPA_HOST = os.getenv("SUPA_HOST_PROD")
SUPA_PORT = os.getenv("SUPA_PORT_PROD")
SUPA_DBNAME = os.getenv("SUPA_DBNAME_PROD")

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
