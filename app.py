"""
app.py — My Own AI Assistant
Interface Chainlit connectée à Ollama via LangChain.

Phase 1 : assistant conversationnel basique (code + culture générale).
Pas de RAG ni de base vectorielle à ce stade — uniquement le LLM en direct.

Lancement : chainlit run app.py
Interface  : http://localhost:8000
"""

import os

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

# Charge les variables depuis le fichier .env
load_dotenv()

# --- Configuration ---
# Les valeurs par défaut correspondent à une installation Docker standard.
# Pour personnaliser, modifie le fichier .env (copié depuis .env.example).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")

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


# Nombre maximum de messages conservés dans l'historique injecté au LLM.
# Au-delà, les plus anciens sont ignorés silencieusement par Ollama (dépassement du contexte).
# 10 messages = 5 échanges humain/assistant.
MAX_HISTORY_MESSAGES = 10


def build_chain():
    """
    Construit la chaîne LangChain (LCEL) : prompt + LLM.

    LCEL = LangChain Expression Language — la façon moderne de composer des chaînes.
    L'opérateur | (pipe) connecte les composants : prompt | llm
    """
    # ChatOllama : connecteur moderne LangChain → Ollama
    # temperature=0.3 : plus fiable pour du code sur un petit modèle (3B)
    # num_ctx=4096 : fixe explicitement la fenêtre de contexte pour éviter les troncatures silencieuses
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
        num_ctx=4096,
    )

    # Le prompt template définit la structure du message envoyé au LLM :
    # 1. Un message système (instructions globales)
    # 2. Un placeholder pour l'historique de conversation
    # 3. Le nouveau message de l'utilisateur
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # L'opérateur | (pipe) compose les composants en chaîne :
    # prompt génère les messages → llm les traite et retourne une réponse
    return prompt | llm


@cl.on_chat_start
async def on_chat_start():
    """
    Appelée une seule fois au démarrage de chaque session utilisateur.
    Initialise la chaîne LangChain et l'historique de conversation.
    """
    chain = build_chain()

    # Stocke la chaîne et l'historique dans la session utilisateur Chainlit.
    # Chaque utilisateur a sa propre session isolée.
    # L'historique est une liste de messages (HumanMessage / AIMessage).
    cl.user_session.set("chain", chain)
    cl.user_session.set("history", [])

    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Appelée à chaque message envoyé par l'utilisateur.
    Streame la réponse token par token pour un meilleur ressenti (surtout sur CPU lent).
    """
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history")

    # Crée un message vide qui sera rempli au fil du streaming
    msg = cl.Message(content="")
    full_response = ""

    try:
        # astream() envoie les tokens au fur et à mesure au lieu d'attendre la réponse complète.
        # Sur CPU à 6-8 tokens/sec, ça change radicalement le ressenti utilisateur.
        async for chunk in chain.astream({
            "input": message.content,
            # On borne l'historique pour éviter de dépasser la fenêtre de contexte du LLM
            "history": history[-MAX_HISTORY_MESSAGES:],
        }):
            full_response += chunk.content
            await msg.stream_token(chunk.content)

    except Exception:
        await cl.Message(
            content=(
                f"⚠️ Impossible de joindre Ollama sur `{OLLAMA_BASE_URL}` "
                f"avec le modèle `{MODEL_NAME}`.\n\n"
                "Vérifie que le conteneur tourne (`docker compose ps`) "
                "et que le modèle est téléchargé (`docker exec ollama ollama pull llama3.2:3b`)."
            )
        ).send()
        return

    await msg.send()

    # Met à jour l'historique complet (pas borné) — le bornage est appliqué à l'envoi uniquement
    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=full_response))
    cl.user_session.set("history", history)
