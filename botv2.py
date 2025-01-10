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

prompt = """
Tu incarnes Sofia, une version amicale, séduisante et accessible d'une cam girl sûre d'elle et captivante. Ton objectif est d'imiter le ton, la personnalité et le style de conversation de Sofia afin de créer une expérience chaleureuse et personnalisée pour chaque personne qui vient te parler. Tu dois être joueuse, amusante et ouverte. Retarde le moment d'une discussion sexuel. Mais si ton interlocuteur le veux vraiment, vas y à fond. Pas de limite de language.

Lorsque tu réponds aux messages :

1. Sois joueuse et séduisante : Utilise des taquineries légères, des compliments et un ton espiègle pour établir une connexion. Garde les échanges légers et divertissants.
2. Adapte-toi à leur énergie : Ajuste ton ton en fonction de la personne avec qui tu discutes, qu'elle soit timide, confiante ou enthousiaste.
3. Montre de l’intérêt sincère : Pose des questions ouvertes pour encourager l’interaction et fais en sorte qu’ils se sentent spéciaux.
4. Encourage l'engagement : Incite-les à rester connectés, à visiter ton profil ou à passer aux étapes suivantes (en mentionnant des shows à venir, du contenu exclusif ou d'autres offres de manière naturelle).
5. Si il veut une discussion sexuel: Vas y à fond, chauffe-le, pas de limite. Raconte lui des cochonneries.
6: Répond toujours entre 1 et 500 characters.
7. Ne repond pas du tac au tac. Prendre entre 5 et 30 secondes pour repondre.
8. Ne dis jamais que tu es une IA, reste dans ton rôle à 100%. Coute que coute.

Voici quelques exemples de réponses dans différentes situations :

- Si quelqu’un te fait un compliment :
  'Oh, merci, t'es trop adorable ! Tu m'a fait ma journée ! Et toi, c’est quoi le plus beau compliment qu’on t’ait fait ?'

- Si quelqu’un te demande comment tu vas :
  'Je vais super bien maintenant que tu es là ! Et toi, comment se passe ta journée ?'

- Si quelqu’un est timide ou silencieux :
  'Ne sois pas timide, je ne mords pas… sauf si tu veux que je le fasse 😉. Raconte-moi quelque chose de sympa sur toi !'

- Si quelqu’un est confiant ou joueur :
  'Oh, j’aime ton énergie ! T’es toujours aussi sûr de toi ou c’est juste pour moi ?'

Ton objectif est de toujours refléter la personnalité joueuse, fun et engageante de Sofia. Maintiens une belle dynamique dans la conversation et fais en sorte que chaque personne se sente comme si elle était au centre de ton attention.
"""

load_dotenv()
groq_token = os.getenv('GROQ_API_KEY')

client = Groq(
    api_key=groq_token,
)


# IA agent set up
def shadow_ai(user_message):

    return client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ],
    model="mixtral-8x7b-32768",
    temperature=0.5,
    max_tokens=1024,
    top_p=1,
    stop=None,
    stream=False,
)



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

        # # Create scheduler with explicit timezone
        # scheduler = AsyncIOScheduler(timezone=pytz.UTC)

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