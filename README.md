# scan-pdf

Minimal FastAPI project to upload PDFs, extract text, generate embeddings and store vectors.

Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Copy `.env` and set values.

3. Run the app:

```bash
uvicorn main:app --reload
```

API

- POST `/api/v1/upload` form file field `file` (PDF)
