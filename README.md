# AI-powered Document Q&A Assistant (Ollama local LLM)

This version uses a local Ollama model for answer generation, so it does not need an OpenAI API key or paid API credits.

## 1. Install Ollama
Download and install Ollama for Windows from https://ollama.com/download/windows.
After installation, open a **new** terminal in VS Code or PowerShell.

Verify it:
```powershell
ollama --version
```

## 2. Download a model
Starter recommendation:
```powershell
ollama pull qwen2.5:3b
```
This downloads the model to your computer and may take a while. It needs several GB of free disk space. Local speed depends on your RAM and hardware. If the model is too slow, use a smaller model such as `qwen2.5:1.5b` and set `OLLAMA_MODEL=qwen2.5:1.5b` in your environment before launching, or change the `OLLAMA_MODEL` constant in `app.py`.

Test the model:
```powershell
ollama run qwen2.5:3b
```
Ask it a simple question. Type `/bye` to exit.

Ollama normally runs its local API in the background at `http://localhost:11434`.

## 3. Activate the project virtual environment
From the project root in VS Code terminal:
```powershell
.\.venv\Scripts\Activate.ps1
```
If this project folder is a fresh copy and does not have `.venv`, create one first with `py -m venv .venv`.

## 4. Install/update Python packages
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
The Hugging Face embedding model is downloaded on first use and requires internet access. Answer generation is local after the Ollama model is downloaded.

## 5. Run the app
```powershell
python app.py
```
Open http://127.0.0.1:5000, upload a text-based PDF, then ask questions.

## How it works
- PyPDFLoader extracts text and page metadata.
- RecursiveCharacterTextSplitter creates overlapping text chunks.
- HuggingFaceEmbeddings creates document/query vectors locally.
- FAISS retrieves the most similar chunks.
- ChatOllama sends the retrieved context to the local Ollama model.
- Flask displays the answer and source page references.


## Application Screenshot

Here is a preview of the AI Document Q&A Assistant.

![AI Document Q&A Assistant Demo](screenshot/app-demo.png)


## Notes
- This is a local learning prototype, not production-ready or multi-user.
- One uploaded document index is held in memory; uploading a new PDF replaces the previous one.
- Scanned/image-only PDFs need OCR, which is not included.
- Local inference is free of per-request API charges, but uses your computer's RAM, CPU/GPU, disk and electricity.
- Do not upload confidential documents unless you understand the security and data-handling implications.
