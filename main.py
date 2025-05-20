import io
import json
import os
from typing import Optional

import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

def classify_pdf_text(text):
        prompt = f"""
            You are a document classifier. Your tasks are:

            1. Classify the document into one of the following categories:
            - "1040": A U.S. Individual Income Tax Return form, typically titled "Form 1040" and includes income, deductions, and tax liability.
            - "W2": A Wage and Tax Statement provided by employers to employees, titled "Form W-2", and includes wages, taxes withheld, and employer details.
            - "1099": A form series used to report non-employment income (e.g., freelance income, interest, dividends). Examples include Form 1099-MISC or 1099-INT.
            - "ID Card": A personal identification card (e.g., driver's license, state ID, or passport) with name, photo, birth date, and possibly issue/expiration dates.
            - "Handwritten note": A note written by hand. OCR may return garbled or unstructured text.
            - "OTHER": Use this for any document that does not match the above, including:
                - Mentions or summaries of documents (e.g., "This is a Form 1099")
                - Other IRS forms (e.g., "Form 1098", "Form 1095")
                - Blank pages or documents with irrelevant content

            2. Extract the **year the document was issued**.
            - Look for phrases like “Issued: 2021”, “Form 1040 (2021)”, etc.
            - **Do not use expiration years** (e.g., “Expires 2028”).
            - If no issuing year is clearly present, return `null`.

            Important notes:
            - Do not classify a document based only on a mention of a form name. The structure and content must match the actual document type.
            - If the document text looks unclear, unstructured, or seems handwritten, classify it as "Handwritten note".
            - If unsure, choose "OTHER".

            Now classify this document.

            Document:
            \"\"\"{text}\"\"\"

            Return the result strictly in this JSON format, as plain text only (no markdown formatting):

            {{"document_type": "...", "year": ...}}

        """
        response = client.chat.completions.create(
            # model="gpt-4",
            model="gpt-4.1-mini",
            # model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content


@app.post("/classify")
async def schedule_classify_task(file: Optional[UploadFile] = File(None)):
    """Endpoint to classify a document into "w2", "1099int", etc"""

    if not file:
        return {"error": "No file uploaded"}

    content = await file.read()
    pdf_stream = io.BytesIO(content)
    
    pdf_reader = pymupdf.open(stream=pdf_stream, filetype="pdf")

    text = ""
    for page in pdf_reader:
        tp = page.get_textpage_ocr(tessdata="C:\\Program Files\\Tesseract-OCR\\tessdata")
        text += page.get_text(textpage=tp)
        

    result = classify_pdf_text(text)
    if result:
        result = json.loads(result)
        return result
    else:
        return {"document_type": "OTHER", "year": None}
