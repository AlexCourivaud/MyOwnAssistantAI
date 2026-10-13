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


def build_chain():
    """
    Construit la chaîne LangChain (LCEL) : prompt + LLM.

    LCEL = LangChain Expression Language — la façon moderne de composer des chaînes.
    L'opérateur | (pipe) connecte les composants : prompt | llm
    """
    # ChatOllama : connecteur moderne LangChain → Ollama
    # Utilise l'API chat (messages) plutôt que l'API completion (texte brut)
    # temperature=0.7 : équilibre entre créativité et précision (0=déterministe, 1=créatif)
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
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
    Transmet le message à la chaîne LangChain avec l'historique et retourne la réponse.
    """
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history")

    # Affiche un indicateur de chargement pendant la génération
    async with cl.Step(name="Réflexion en cours..."):
        response = await chain.ainvoke({
            "input": message.content,
            "history": history,
        })

    # Met à jour l'historique avec le nouveau tour de conversation
    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=response.content))
    cl.user_session.set("history", history)

    await cl.Message(content=response.content).send()
