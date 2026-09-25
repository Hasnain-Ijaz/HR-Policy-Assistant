# HR Policy Assistant (RAG)

A browser-based HR Policy Q&A app built with Streamlit, FAISS, Sentence Transformers, PyMuPDF, and Groq (`openai/gpt-oss-20b`).

## Features

- Upload a text-based HR policy PDF.
- Extract page-wise text with PyMuPDF.
- Split policy text into overlapping chunks.
- Create semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieve relevant chunks using FAISS cosine similarity.
- Generate grounded answers with Groq's `openai/gpt-oss-20b`.
- Show retrieved excerpts and page numbers.
- Keeps the uploaded document index and chat in the current Streamlit session.

## Project files

```text
hr-policy-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository named `hr-policy-assistant`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore` to the repository root.
3. Create or sign in to a Groq account at https://console.groq.com/ and create an API key.
4. Open https://share.streamlit.io/ and connect your GitHub account.
5. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
6. Before or after deploying, open the app's **Settings → Secrets** and add:

   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

   Do not put the key in `app.py`, GitHub files, screenshots, or public commits.
7. Save secrets and deploy/reboot the app.

## How to use

1. Open the deployed Streamlit URL.
2. Use the sidebar to upload a text-based HR policy PDF.
3. Wait for the document to be extracted and indexed.
4. Ask questions in the chat input.
5. Expand **View retrieved policy sources** to inspect the page excerpts used.

## Important limitations

- Scanned/image-only PDFs are not supported because OCR is not included. Use a searchable PDF.
- The app has a 25 MB upload limit and 500-page limit by default.
- The embedding model downloads the first time the app starts; the initial launch can take longer.
- The app sends the question and retrieved policy excerpts to Groq. Do not upload sensitive or confidential HR documents unless you are authorized and have reviewed applicable policies.
- RAG can miss relevant passages or produce errors. Check the cited policy pages and confirm sensitive matters with HR.
- This is a demo app, not an official HR decision system.

## Model configuration

- Chat model: `openai/gpt-oss-20b` (Groq)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector search: FAISS `IndexFlatIP` with normalized vectors (cosine similarity)

## Troubleshooting

### Missing Groq API key
Add `GROQ_API_KEY` under Streamlit Cloud → app Settings → Secrets, save, then reboot the app.

### Invalid API key / authentication error
Create a new key in the Groq console and replace the value in Streamlit Secrets. Keep quotes around the value.

### Model not found or unavailable
Confirm that `openai/gpt-oss-20b` is available to your Groq account. If Groq changes model availability, update `GROQ_MODEL` in `app.py` to a currently supported model.

### App fails during dependency installation
Check the Streamlit Cloud build logs. This project uses `faiss-cpu`; use a supported Python version in Streamlit Cloud (Python 3.11 is a practical choice if the deployment settings offer it).

### No text extracted
The PDF may be scanned or image-only. This version does not include OCR; upload a searchable/text-based PDF.

### Answers are incomplete
Try a more specific question, check the retrieved excerpts, or adjust `TOP_K`, `CHUNK_SIZE`, and `CHUNK_OVERLAP` in `app.py`.
