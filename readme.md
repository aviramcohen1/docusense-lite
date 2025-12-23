סגור. הנה **README מוכן, ברור, כן, ולא מתנצל**.
אתה יכול להעתיק כמו שהוא ל־`README.md`.

---

# DocuSense Lite – Intelligent Ticket Classification Backend

## Overview

**DocuSense Lite** is a backend service that automatically classifies customer support tickets based on their textual content.

The system ingests raw support tickets from a CSV file, processes the text, applies a machine-learning model to predict the ticket category, stores the results in a local database, and exposes a clean REST API for retrieval.

The project demonstrates practical use of:

* Text processing
* Machine learning
* Backend API design
* Data persistence

All components run locally and are easy to understand and extend.

---

## Real-World Problem

Customer support teams receive large volumes of free-text tickets such as:

* “I was charged twice”
* “The app crashes when I open settings”
* “My package hasn’t arrived yet”

Manually reading and routing these tickets:

* Takes time
* Is error-prone
* Does not scale

**DocuSense Lite** automates the first decision:

> *What is this ticket about?*

---

## What the System Does

1. Accepts a CSV file containing support tickets
2. Cleans and combines the textual fields
3. Applies a machine-learning model to classify each ticket
4. Stores the ticket and prediction in a SQLite database
5. Exposes API endpoints to retrieve the results

---

## Example Output

After ingesting a CSV file, the system returns data such as:

```json
{
  "subject": "Charged twice",
  "body": "I was charged twice for the same invoice",
  "predicted_category": "Billing",
  "created_at": "2025-12-23T19:40:30"
}
```

This proves that the system:

* Understands unstructured text
* Produces an automated decision
* Persists the result
* Serves it via an API

---

## Architecture (High Level)

```
CSV File
   ↓
Text Cleaning & Normalization
   ↓
TF-IDF Feature Extraction
   ↓
Logistic Regression Classifier
   ↓
SQLite Database
   ↓
FastAPI REST API
```

---

## Tech Stack

* **Python 3**
* **FastAPI** – REST API and Swagger UI
* **scikit-learn** – ML pipeline (TF-IDF + Logistic Regression)
* **Pandas** – CSV ingestion
* **SQLModel / SQLite** – Data persistence
* **Uvicorn** – ASGI server

---

## API Endpoints

### `POST /ingest/csv`

Upload a CSV file containing tickets.

**Required columns:**

* `subject`
* `body`

---

### `GET /tickets`

Retrieve stored tickets and their predicted categories.

---

### `GET /tickets/{id}`

Retrieve a single ticket by ID.

---

## Running the Project Locally

```bash
pip install fastapi uvicorn pandas scikit-learn sqlmodel python-multipart
python -m uvicorn app:app
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## Why This Project Matters

This project demonstrates the ability to:

* Convert unstructured text into structured decisions
* Apply machine learning to real business problems
* Design clean, maintainable backend services
* Build systems that replace manual human judgment with automation

---

## Possible Extensions

* Confidence scores for predictions
* Priority classification (Low / Medium / High)
* Duplicate ticket detection
* Integration with external ticketing systems
* OCR support for image-based tickets

---

## Author’s Note

This project focuses on clarity, correctness, and real-world relevance rather than over-engineering.
It is intentionally simple, transparent, and interview-friendly.

---

