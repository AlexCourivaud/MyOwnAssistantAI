"""
app.py — My Own AI Assistant
Interface Chainlit connectée à Ollama via LangChain.

Phase 2 : mémoire persistante des conversations via SQLite.
Phase 3 : RAG sur documents personnels via ChromaDB.

Le routage RAG / LLM direct se fait sans appel au LLM : on cherche dans ChromaDB
et on utilise le RAG seulement si des extraits sont suffisamment proches de la question.

Lancement : chainlit run app.py
Interface  : http://localhost:8000
"""

import logging
import os
import time

import chainlit as cl
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Charge les variables depuis le fichier .env
load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")

# Durée pendant laquelle Ollama garde le modèle en mémoire après un appel.
# Évite le rechargement (plusieurs secondes) après une pause.
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m")

# Nombre maximum de tokens générés par réponse (borne le temps de génération).
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "700"))

# Chemin vers la base SQLite qui stocke l'historique des conversations.
DB_PATH = os.getenv("DB_PATH", "chat_history.db")

# Identifiant fixe de la conversation stockée dans SQLite.
SESSION_ID = os.getenv("SESSION_ID", "default")

# Dossier ChromaDB généré par ingest.py
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# Modèle d'embeddings — lu depuis .env pour garantir la cohérence avec ingest.py.
# Les deux fichiers doivent utiliser le même modèle, sinon les vecteurs sont incompatibles.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Nombre de chunks récupérés depuis ChromaDB à chaque requête RAG
RAG_TOP_K = 3

# Seuil de distance au-delà duquel un extrait est jugé hors sujet (plus petit = plus proche).
# Valeur de départ à CALIBRER avec les logs : voir les lignes "Retrieval" dans la console.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "1.0"))

# Historique envoyé au LLM : plus court en mode RAG, car les extraits alourdissent le prompt.
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_MESSAGES_RAG = 4

# Message affiché au démarrage de chaque session
WELCOME_MESSAGE = (
    "👋 Bonjour ! Je suis ton assistant personnel.\n\n"
    "Je peux t'aider sur :\n"
    "- 💻 Des questions de **code** (Python, Docker, Git...)\n"
    "- 🧠 Des questions de **culture générale** et concepts techniques\n"
    "- 📄 Des questions sur tes **documents personnels** (si tu as lancé `ingest.py`)\n\n"
    "Pose-moi ta première question !"
)

# Prompt système : définit le comportement général de l'assistant
SYSTEM_PROMPT = (
    "Tu es un assistant personnel utile et précis. "
    "Réponds toujours en français, même si les extraits ou la question contiennent de l'anglais. "
    "Garde les termes techniques courants (decorator, commit, container...) tels quels. "
    "Tu es concis mais complet dans tes réponses."
)

# Prompt RAG : utilisé quand des extraits de documents sont disponibles.
# {context} sera remplacé par les chunks récupérés depuis ChromaDB.
RAG_SYSTEM_PROMPT = (
    "Tu es un assistant personnel utile et précis. "
    "Réponds toujours en français, même si les extraits ou la question contiennent de l'anglais. "
    "Garde les termes techniques courants (decorator, commit, container...) tels quels. "
    "Utilise les extraits de documents ci-dessous pour répondre à la question. "
    "Si la réponse ne se trouve pas dans les extraits, dis-le clairement. "
    "Reste concis.\n\n"
    "Extraits de documents :\n{context}"
)


def build_llm() -> ChatOllama:
    """
    Instancie le LLM Ollama partagé par toutes les chaînes.

    temperature=0.3 : plus fiable pour du code sur un petit modèle (3B).
    num_ctx=4096 : fixe explicitement la fenêtre de contexte.
    num_predict : borne la longueur de la réponse.
    keep_alive : garde le modèle chargé en mémoire entre deux questions.
    """
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
        num_ctx=4096,
        num_predict=NUM_PREDICT,
        keep_alive=KEEP_ALIVE,
    )


def build_chain(llm: ChatOllama):
    """
    Chaîne LCEL standard : prompt + LLM (questions générales, sans RAG).
    La mémoire est gérée séparément via SQLChatMessageHistory.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return prompt | llm


def build_rag_chain(llm: ChatOllama):
    """
    Chaîne RAG : prompt enrichi avec le contexte docs + LLM.
    Le contexte ({context}) est injecté avant l'appel.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return prompt | llm


def load_vectorstore() -> Chroma | None:
    """
    Charge la base vectorielle ChromaDB depuis le disque.

    Retourne None si chroma_db/ n'existe pas encore
    (l'utilisateur n'a pas encore lancé ingest.py).
    """
    if not os.path.exists(CHROMA_DIR):
        logger.warning(
            "ChromaDB introuvable dans '%s'. Lance 'python ingest.py' pour indexer tes documents.",
            CHROMA_DIR,
        )
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """
    Retourne l'historique SQLite pour une conversation donnée.

    Args:
        session_id: identifiant de la conversation (voir SESSION_ID)
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=f"sqlite:///{DB_PATH}",
    )


# --- Ressources lourdes : chargées UNE SEULE FOIS au démarrage du serveur ---
# Avant, le modèle d'embeddings (PyTorch) était rechargé à chaque nouvelle session,
# ce qui coûtait plusieurs secondes à chaque rafraîchissement de la page.
LLM = build_llm()
CHAIN = build_chain(LLM)
RAG_CHAIN = build_rag_chain(LLM)
VECTORSTORE = load_vectorstore()


@cl.on_chat_start
async def on_chat_start():
    """
    Appelée au démarrage de chaque session utilisateur.
    Charge l'historique persistant depuis SQLite.
    """
    chat_history = get_session_history(SESSION_ID)
    cl.user_session.set("chat_history", chat_history)

    # make_async : exécute l'appel SQLite bloquant sans figer l'interface
    existing_messages = await cl.make_async(lambda: chat_history.messages)()
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
    Cherche dans ChromaDB : si des extraits sont assez proches, passe par le RAG,
    sinon répond directement. Streame la réponse et sauvegarde l'échange dans SQLite.
    """
    chat_history = cl.user_session.get("chat_history")

    msg = cl.Message(content="")
    full_response = ""
    docs = []

    try:
        # --- Retrieval (sans appel au LLM) ---
        if VECTORSTORE is not None:
            async with cl.Step(name="Recherche dans tes documents"):
                t0 = time.perf_counter()
                # make_async : le calcul d'embedding (CPU) ne bloque pas l'interface
                results = await cl.make_async(VECTORSTORE.similarity_search_with_score)(
                    message.content, k=RAG_TOP_K
                )
                logger.info(
                    "Retrieval %.2fs — distances : %s (seuil %.2f)",
                    time.perf_counter() - t0,
                    [round(dist, 3) for _, dist in results],
                    RAG_MAX_DISTANCE,
                )
                # On ne garde que les extraits assez proches de la question
                docs = [doc for doc, dist in results if dist <= RAG_MAX_DISTANCE]

        use_rag = bool(docs)

        # Borne l'historique envoyé au LLM (plus court en mode RAG)
        all_messages = await cl.make_async(lambda: chat_history.messages)()
        max_history = MAX_HISTORY_MESSAGES_RAG if use_rag else MAX_HISTORY_MESSAGES
        recent_messages = all_messages[-max_history:]

        if use_rag:
            chain = RAG_CHAIN
            inputs = {
                "input": message.content,
                "history": recent_messages,
                "context": "\n\n---\n\n".join(doc.page_content for doc in docs),
            }
        else:
            chain = CHAIN
            inputs = {"input": message.content, "history": recent_messages}

        # --- Génération en streaming ---
        t1 = time.perf_counter()
        first_token = True
        async for chunk in chain.astream(inputs):
            if first_token:
                logger.info(
                    "Premier token après %.2fs (mode %s)",
                    time.perf_counter() - t1,
                    "RAG" if use_rag else "direct",
                )
                first_token = False
            full_response += chunk.content
            await msg.stream_token(chunk.content)
        logger.info("Génération totale %.2fs", time.perf_counter() - t1)

    except Exception:
        # Le détail technique va dans les logs, l'utilisateur voit un message clair
        logger.exception("Erreur lors du traitement de la question")
        await cl.Message(
            content=(
                f"⚠️ Une erreur est survenue (Ollama : `{OLLAMA_BASE_URL}`, "
                f"modèle `{MODEL_NAME}`).\n\n"
                "Vérifie que Ollama tourne (`docker compose ps`) "
                "et que le modèle est téléchargé "
                f"(`docker compose exec ollama ollama pull {MODEL_NAME}`). "
                "Le détail de l'erreur est dans la console."
            )
        ).send()
        return

    await msg.send()

    # Affiche les sources utilisées si la réponse vient du RAG
    if docs:
        sources = sorted({doc.metadata.get("source", "source inconnue") for doc in docs})
        sources_text = "\n".join(f"  - `{s}`" for s in sources)
        await cl.Message(content=f"📄 **Sources utilisées :**\n{sources_text}").send()

    # Sauvegarde l'échange dans SQLite — persistant même après redémarrage
    await cl.make_async(chat_history.add_message)(HumanMessage(content=message.content))
    await cl.make_async(chat_history.add_message)(AIMessage(content=full_response))
