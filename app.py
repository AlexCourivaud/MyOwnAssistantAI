"""
app.py — My Own AI Assistant
Interface Chainlit connectée à Ollama via LangChain.

Phase 2 : mémoire persistante des conversations via SQLite.
L'historique survit aux redémarrages de l'application.

Lancement : chainlit run app.py
Interface  : http://localhost:8000
"""

import logging
import os

import chainlit as cl
from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

# Charge les variables depuis le fichier .env
load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")

# Chemin vers la base SQLite qui stocke l'historique des conversations.
# Le fichier est créé automatiquement s'il n'existe pas.
DB_PATH = os.getenv("DB_PATH", "chat_history.db")

# Identifiant de la conversation stockée dans SQLite.
# Fixe, pour retrouver l'historique après un redémarrage ou un rafraîchissement.
# Changer cette valeur démarre une nouvelle conversation (l'ancienne reste en base).
SESSION_ID = os.getenv("SESSION_ID", "default")

# Nombre maximum de messages injectés dans le contexte du LLM à chaque appel.
# L'historique complet reste en base — on limite seulement ce qu'on envoie au LLM.
MAX_HISTORY_MESSAGES = 10

# Message affiché au démarrage de chaque session
WELCOME_MESSAGE = (
    "👋 Bonjour ! Je suis ton assistant personnel.\n\n"
    "Je peux t'aider sur :\n"
    "- 💻 Des questions de **code** (Python, Docker, Git...)\n"
    "- 🧠 Des questions de **culture générale** et concepts techniques\n\n"
    "Pose-moi ta première question !"
)

# Prompt système : définit le comportement général de l'assistant
SYSTEM_PROMPT = (
    "Tu es un assistant personnel utile et précis. "
    "Tu réponds en français par défaut sauf si on te parle dans une autre langue. "
    "Tu es concis mais complet dans tes réponses."
)


def build_chain():
    """
    Construit la chaîne LangChain (LCEL) : prompt + LLM.
    La mémoire est gérée séparément via SQLChatMessageHistory.
    """
    # temperature=0.3 : plus fiable pour du code sur un petit modèle (3B)
    # num_ctx=4096 : fixe explicitement la fenêtre de contexte
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
        num_ctx=4096,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    return prompt | llm


def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """
    Retourne l'historique SQLite pour une conversation donnée.

    SQLChatMessageHistory stocke chaque message dans une table SQLite.
    Chaque session_id correspond à une conversation distincte.
    Si la conversation n'existe pas encore, elle est créée automatiquement.

    Args:
        session_id: identifiant de la conversation (voir SESSION_ID)
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=f"sqlite:///{DB_PATH}",
    )


@cl.on_chat_start
async def on_chat_start():
    """
    Appelée au démarrage de chaque session utilisateur.
    Initialise la chaîne et charge l'historique persistant depuis SQLite.
    """
    chain = build_chain()

    # Identifiant fixe : l'assistant est mono-utilisateur, donc toutes les
    # sessions Chainlit partagent la même conversation persistante.
    chat_history = get_session_history(SESSION_ID)

    cl.user_session.set("chain", chain)
    cl.user_session.set("chat_history", chat_history)

    # Vérifie si des messages existent déjà pour cette conversation
    existing_messages = chat_history.messages
    if existing_messages:
        await cl.Message(
            content=f"🔄 Conversation restaurée ({len(existing_messages) // 2} échanges précédents)."
        ).send()
    else:
        await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Appelée à chaque message utilisateur.
    Streame la réponse et sauvegarde l'échange dans SQLite.
    """
    chain = cl.user_session.get("chain")
    chat_history = cl.user_session.get("chat_history")

    # Récupère tous les messages depuis SQLite, puis on borne ce qu'on envoie
    # au LLM pour rester dans la fenêtre de contexte.
    all_messages = chat_history.messages
    recent_messages = all_messages[-MAX_HISTORY_MESSAGES:]

    msg = cl.Message(content="")
    full_response = ""

    try:
        async for chunk in chain.astream({
            "input": message.content,
            "history": recent_messages,
        }):
            full_response += chunk.content
            await msg.stream_token(chunk.content)

    except Exception:
        # Le détail technique va dans les logs, l'utilisateur voit un message clair
        logger.exception("Erreur lors de l'appel à Ollama")
        await cl.Message(
            content=(
                f"⚠️ Impossible de joindre Ollama sur `{OLLAMA_BASE_URL}` "
                f"avec le modèle `{MODEL_NAME}`.\n\n"
                "Vérifie que le conteneur tourne (`docker compose ps`) "
                "et que le modèle est téléchargé "
                f"(`docker compose exec ollama ollama pull {MODEL_NAME}`)."
            )
        ).send()
        return

    await msg.send()

    # Sauvegarde l'échange dans SQLite — persistant même après redémarrage
    chat_history.add_message(HumanMessage(content=message.content))
    chat_history.add_message(AIMessage(content=full_response))