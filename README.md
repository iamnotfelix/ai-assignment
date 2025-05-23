# AI Assignment

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Tesseract OCR installed on your system
- OpenAI API key

## Setup

0. Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Clone this repo
2. Install Tesseract OCR from [here](https://github.com/UB-Mannheim/tesseract/wiki) if not installed already.
3. Create a `.env` file in the root directory with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

4. Install dependencies using uv:

```bash
uv sync
```

5. Run:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

6. You can now open the docs at http://0.0.0.0:8000/docs and run the endpoints

## API Endpoints

### Docs

http://0.0.0.0:8000/docs

### Document Classification

- `POST /classify` - Submit a document to be classified

## Task

Your task is to complete the /classify endpoint
The endpoint should

1. Take in a PDF file as an input - Use the sample documents provided under sample directory
2. Classify the PDF as one of
   - "1040"
   - "W2"
   - "1099"
   - "ID Card"
   - "Handwritten note"
   - "OTHER"
3. Also parse the year the document was issued

## Troubleshooting

1. If you get Tesseract-related errors:

   - Verify Tesseract is installed correctly
   - Ensure the path is correct: `C:\\Program Files\\Tesseract-OCR\\tessdata`

2. If you get OpenAI API errors:

   - Verify your API key is correctly set in the `.env` file
   - Check if you have sufficient credits in your OpenAI account
