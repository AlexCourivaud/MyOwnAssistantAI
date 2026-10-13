# 🤖 My Own AI Assistant

> A fully local, private AI assistant that runs entirely on your machine.
> No data sent to third parties. Free. Open source.

---

## ✨ Features

- 💬 Conversational assistant for code and general knowledge questions
- 🔒 100% local — your data never leaves your machine
- 🧠 Powered by Llama 3.2 3B via Ollama
- 🌐 Web interface via Chainlit
- 📄 Document-aware Q&A with RAG *(coming in Phase 3)*

---

## 🏗️ Architecture

```
User (Chainlit UI)
      ↓
LangChain (orchestration)
      ↓
Ollama — Llama 3.2 3B (local LLM)
      ↓
Response
```

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| [Ollama](https://ollama.com) | Run LLM locally |
| [Llama 3.2 3B](https://ollama.com/library/llama3.2) | Language model (Meta) |
| [LangChain](https://langchain.com) | LLM orchestration |
| [Chainlit](https://chainlit.io) | Web chat interface |
| [Docker Compose](https://docs.docker.com/compose/) | Service deployment |

---

## 🚀 Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Python 3.10+
- ~3 GB free disk space (for the model)

### 1. Clone the repository

```bash
git clone https://github.com/AlexCourivaud/my-own-ai-assistant.git
cd my-own-ai-assistant
```

### 2. Start Ollama

```bash
docker compose up -d
```

### 3. Download the model (first time only)

```bash
docker exec ollama ollama pull llama3.2:3b
```

### 4. Set up Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 6. Launch the assistant

```bash
chainlit run app.py
```

Open your browser at **http://localhost:8000** 🎉

---

## 📁 Project Structure

```
my-own-ai-assistant/
├── docker-compose.yml   # Ollama service
├── app.py               # Chainlit assistant
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

---

## 🗺️ Roadmap

- [x] Phase 1 — Local assistant (code & general knowledge)
- [ ] Phase 2 — Persistent conversation memory
- [ ] Phase 3 — RAG on personal documents (ChromaDB)
- [ ] Phase 4 — Private network access (multi-device)
- [ ] Phase 5 — Agents, tools, LangSmith observability

---

## 📄 License

MIT — feel free to use, modify and share.
