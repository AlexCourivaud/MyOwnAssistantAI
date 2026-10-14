"""
ingest.py — My Own AI Assistant
Script d'ingestion des documents dans ChromaDB.

Charge les fichiers du dossier documents/ (PDF, TXT, MD),
les découpe en chunks, calcule les embeddings et les stocke
dans la base vectorielle ChromaDB.

À lancer une première fois, puis à chaque ajout ou modification de documents :
    python ingest.py

La base existante est supprimée puis reconstruite à chaque lancement,
ce qui évite les doublons et le mélange de modèles d'embeddings.

Phase 3 : RAG sur documents personnels.
"""

import logging
import os
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Charge le fichier .env : sans ça, EMBEDDING_MODEL et CHROMA_DIR seraient ignorés
# et ingest.py pourrait utiliser un autre modèle qu'app.py.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---

# Dossier contenant tes documents (PDF, TXT, MD)
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")

# Dossier où ChromaDB stocke la base vectorielle sur le disque
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# Modèle d'embeddings multilingue (français inclus).
# DOIT être identique dans app.py : on le lit donc depuis .env dans les deux fichiers.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Taille d'un chunk en caractères (1000 caractères ≈ 250 tokens en français).
# Ce modèle ne lit qu'environ 128 tokens (~500 caractères) : au-delà, la fin du chunk
# est ignorée par l'embedding. On reste donc autour de 500 caractères.
# Overlap : chevauchement entre chunks consécutifs pour préserver le contexte.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))


def load_documents(directory: str):
    """
    Charge tous les fichiers supportés depuis le dossier documents/.

    Supporte : PDF (.pdf), texte brut (.txt), Markdown (.md)

    Args:
        directory: chemin vers le dossier contenant les documents
    """
    loaders = [
        # PDF : chaque page devient un document LangChain
        DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
        # TXT : chaque fichier devient un document LangChain
        DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        ),
        # MD : traité comme du texte brut
        DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        ),
    ]

    all_docs = []
    for loader in loaders:
        docs = loader.load()
        all_docs.extend(docs)
        if docs:
            logger.info("Chargé %d document(s) via %s", len(docs), loader.__class__.__name__)

    return all_docs


def split_documents(documents):
    """
    Découpe les documents en chunks pour le RAG.

    RecursiveCharacterTextSplitter essaie de couper sur les séparateurs naturels
    (paragraphes, phrases) avant de couper brutalement à la taille limite.

    Args:
        documents: liste de documents LangChain chargés depuis load_documents()
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Séparateurs essayés dans l'ordre : paragraphe > ligne > phrase > mot
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def main():
    logger.info("=== Démarrage de l'ingestion ===")

    # 1. Chargement des documents
    logger.info("Chargement des documents depuis '%s'...", DOCUMENTS_DIR)
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        logger.warning("Aucun document trouvé dans '%s'. Ajoute des fichiers PDF, TXT ou MD.", DOCUMENTS_DIR)
        return

    logger.info("Total : %d document(s) chargé(s)", len(documents))

    # 2. Découpage en chunks
    logger.info("Découpage en chunks (taille=%d, overlap=%d)...", CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = split_documents(documents)
    logger.info("Total : %d chunk(s) créé(s)", len(chunks))

    # 3. Initialisation du modèle d'embeddings
    # Le modèle est téléchargé automatiquement au premier lancement
    # et mis en cache dans ~/.cache/huggingface/
    # normalize_embeddings=True : vecteurs normalisés, donc des distances comparables
    # et un seuil de pertinence plus facile à régler. Même réglage dans app.py.
    logger.info("Chargement du modèle d'embeddings '%s'...", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. Suppression de l'ancienne base
    # Chroma.from_documents() AJOUTE à la collection existante (il n'écrase rien) :
    # relancer le script dupliquerait les chunks, et un changement de modèle
    # d'embeddings mélangerait des vecteurs incompatibles sans erreur visible.
    if os.path.exists(CHROMA_DIR):
        logger.info("Suppression de l'ancienne base '%s'...", CHROMA_DIR)
        shutil.rmtree(CHROMA_DIR)

    # 5. Stockage dans ChromaDB
    logger.info("Stockage dans ChromaDB ('%s')...", CHROMA_DIR)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    logger.info("=== Ingestion terminée — %d chunks stockés dans '%s' ===", len(chunks), CHROMA_DIR)


if __name__ == "__main__":
    main()