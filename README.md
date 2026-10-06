# Drive RAG Assistant

RAG assistant that answers questions from Google Drive documents. Loads files via a service account, chunks and embeds them with Gemini, stores vectors in ChromaDB, retrieves the top matches, and generates concise, grounded answers with guardrails against hallucination and prompt injection.

## How it works

1. **Load**: [drive_loader.py](drive_loader.py) reads every supported file in a Drive folder (Google Docs, Google Sheets, PDF, TXT, CSV, Markdown).
2. **Chunk**: [chunking.py](chunking.py) splits each document into overlapping chunks.
3. **Embed and store**: chunks are embedded with `gemini-embedding-001` and saved in a local ChromaDB collection.
4. **Retrieve and answer**: the question is embedded, the top 5 chunks are retrieved, and a Gemini model answers using only that context.

The index is built on the first run and reused afterwards. Delete the `chroma_db` folder to re-index after your Drive files change.

## Setup

1. Install dependencies (Python 3.10+):

   ```bash
   pip install chromadb python-dotenv google-genai google-api-python-client google-auth pypdf
   ```

2. Create a Google Cloud service account, download its JSON key, and share your Drive folder with the service account's email (Viewer access).
3. Create a `.env` file:

   ```
   GEMINI_API_KEY=your-gemini-api-key
   GOOGLE_SERVICE_ACCOUNT_FILE=path/to/service-account.json
   DRIVE_FOLDER_ID=your-drive-folder-id
   ```

4. Run:

   ```bash
   python rag_system.py
   ```

Type a question at the prompt, or `exit` to quit.

## Notes

- Never commit `.env` or the service account key. Both are listed in `.gitignore`.
- API calls retry automatically with exponential backoff on 429/500/503/504 errors.

## License

[MIT](LICENSE)
