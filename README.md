# Local_ai_agent
this is the tool for my personal use 


# 🤖 Local AI Agent — RAG-Based Question Answering System

A locally running AI Agent built with **Python, Ollama, Embeddings, and ChromaDB**, using **Retrieval-Augmented Generation (RAG)** to answer questions from a custom knowledge base (coffee/restaurant CSV + PDF data).

Instead of relying only on an LLM's built-in knowledge, this project retrieves relevant information from your own documents and gives that as context to the model before it answers.

---

## 🧠 How It Works

```
Documents (CSV/PDF)
        ↓
Chunking + Embeddings
        ↓
ChromaDB (Vector Store)
        ↓
User Question → Embedding → Similarity Search → Relevant Chunks
        ↓
Question + Relevant Chunks → Ollama LLM
        ↓
Final Answer
```

1. Your CSV/PDF data is split into chunks and converted into embeddings.
2. Embeddings are stored in **ChromaDB**.
3. When you ask a question, it's converted to an embedding and matched against the stored chunks.
4. The most relevant chunks are sent to a local LLM (via **Ollama**) along with your question.
5. The LLM answers using that retrieved context.

---

## 🛠️ Tech Stack

- **Python** – core language
- **Ollama** – runs the LLM locally (Llama, Mistral, etc.)
- **Embedding model** – converts text to vectors (e.g. `nomic-embed-text`)
- **ChromaDB** – vector database for semantic search
- **LangChain** – ties the RAG pipeline together

---

## 📂 Project Structure

```
Local_ai_agent/
├── main.py                                   # Main app: takes questions, retrieves context, generates answers
├── vector.py                                  # Embeddings + ChromaDB setup/retrieval
├── test_embedding.py                          # Quick test for the embedding model
├── requirements.txt                           # Python dependencies
├── coffee_restaurant_large_dataset.csv        # Sample structured data
├── coffee_restaurant_product_reviews_5000.pdf # Sample unstructured data
└── .gitignore
```

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/SatyamRaj2007/Local_ai_agent.git
cd Local_ai_agent

# 2. Create & activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama, then pull a model + embedding model
ollama pull llama3.2
ollama pull nomic-embed-text
```

---

## ▶️ Usage

```bash
# Run the app
python main.py

# Test that embeddings are working
python test_embedding.py
```

### Example Questions
- What coffee products are available?
- Which coffee has the highest rating?
- What do customers say about the espresso?
- What are the common complaints about the products?

---

## ⚠️ Limitations

- Answer quality depends on chunking and retrieval quality.
- The LLM can still hallucinate, especially with insufficient context.
- Running LLMs locally needs decent CPU/GPU/RAM.
- This is a **learning prototype**, not production-ready (no auth, testing, or monitoring yet).

---

## 🔐 Security Note

Never commit secrets. `.gitignore` already excludes `.venv/`, `.env`, `__pycache__/`, and the generated `chroma_langchain_db/` folder (it's rebuilt automatically from your source documents).

---

## 🚀 Future Improvements

- Web-based chat interface
- Conversational memory
- Source citations in answers
- Streaming responses
- Support for more file formats (TXT, DOCX, JSON)

---

## 👨‍💻 Author

**Satyam Raj** — Engineering student exploring AI, ML, Generative AI, RAG, and full-stack development.

⭐ If this project helped you, consider starring the repo on GitHub.

## 📜 License

For educational and learning purposes.