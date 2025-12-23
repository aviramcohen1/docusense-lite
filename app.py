from datetime import datetime
from io import StringIO
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlmodel import SQLModel, Field, Session, create_engine, select

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------
# DB
# -----------------------
class Ticket(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    subject: str
    body: str
    combined_text: str
    predicted_category: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

engine = create_engine("sqlite:///docusense.db", echo=False)
SQLModel.metadata.create_all(engine)


# -----------------------
# ML (simple demo model)
# -----------------------
def build_demo_model() -> Pipeline:
    # A tiny demo model trained on small in-code examples so the app works immediately.
    train_texts = [
        "cannot login password reset account locked",
        "invoice charged twice refund billing",
        "app crashes when opening settings technical bug",
        "delivery late package not arrived shipping",
        "change email update profile account",
        "payment failed credit card billing",
        "error 500 on checkout technical",
        "where is my order delivery tracking",
    ]
    train_labels = [
        "Account", "Billing", "Technical", "Delivery",
        "Account", "Billing", "Technical", "Delivery"
    ]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(train_texts, train_labels)
    return pipe

MODEL = build_demo_model()


# -----------------------
# API
# -----------------------
app = FastAPI(title="DocuSense Lite", version="0.1.0")


@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = (await file.read()).decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(raw))

    required = {"subject", "body"}
    if not required.issubset(set(df.columns.str.lower())):
        raise HTTPException(status_code=400, detail="CSV must include columns: subject, body")

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    created_ids: List[int] = []

    with Session(engine) as session:
        for _, row in df.iterrows():
            subject = str(row.get("subject", "")).strip()
            body = str(row.get("body", "")).strip()
            combined = f"{subject} {body}".strip()

            pred = MODEL.predict([combined])[0]

            t = Ticket(
                subject=subject,
                body=body,
                combined_text=combined,
                predicted_category=str(pred),
            )
            session.add(t)
            session.commit()
            session.refresh(t)
            created_ids.append(t.id)

    return {"ingested": len(created_ids), "ticket_ids": created_ids[:20]}


@app.get("/tickets")
def list_tickets(
    limit: int = 50,
    category: str | None = None
):
    with Session(engine) as session:
        query = select(Ticket)

        if category:
            query = query.where(Ticket.predicted_category == category)

        query = query.order_by(Ticket.id.desc()).limit(limit)
        items = session.exec(query).all()
        return items


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int):
    with Session(engine) as session:
        t = session.get(Ticket, ticket_id)
        if not t:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return t
