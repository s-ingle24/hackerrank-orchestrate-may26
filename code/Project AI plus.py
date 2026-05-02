import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# PATHS

TICKETS_PATH = "E:/Hacker Rank/support_tickets/support_tickets.csv"
OUTPUT_PATH = "E:/Hacker Rank/support_tickets/output.csv"
DATA_DIR = "E:/Hacker Rank/data"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# LOAD DOCUMENTS

def load_documents():
    docs = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".txt"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                        docs.append(f.read())
                except:
                    pass
    print("Documents loaded:", len(docs))
    return docs

DOCUMENTS = load_documents()


# CLEAN TEXT

def clean(text):
    return re.sub(r'[^a-z0-9 ]', ' ', str(text).lower())


# TF-IDF MODEL

vectorizer = TfidfVectorizer(stop_words="english")

if DOCUMENTS:
    DOC_VECTORS = vectorizer.fit_transform([clean(d) for d in DOCUMENTS])
else:
    DOC_VECTORS = None


# RETRIEVAL (SMART)

def retrieve_docs(ticket, top_k=2):
    if DOC_VECTORS is None:
        return []

    ticket_vec = vectorizer.transform([clean(ticket)])
    sims = cosine_similarity(ticket_vec, DOC_VECTORS).flatten()

    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for i in top_idx:
        if sims[i] > 0.05:
            results.append(DOCUMENTS[i][:300])

    return results


# CLASSIFICATION

def classify(ticket):
    t = ticket.lower()

    if any(w in t for w in ["bug", "crash", "error", "exception"]):
        return "bug"

    elif any(w in t for w in ["feature", "suggest", "add", "improve"]):
        return "feature_request"

    else:
        return "product_issue"


# PRODUCT AREA DETECTION

def detect_area(ticket):
    t = ticket.lower()

    if any(w in t for w in ["payment","refund","charged","money","transaction","visa","card"]):
        return "payments"

    elif any(w in t for w in ["login","password","account","access","workspace"]):
        return "authentication"

    elif any(w in t for w in ["test","score","submission","recruiter","hackerank"]):
        return "coding_platform"

    elif "api" in t:
        return "api"

    else:
        return "general"


# RISK DETECTION

def is_high_risk(ticket):
    t = ticket.lower()

    if any(w in t for w in ["payment","money","charged","refund"]):
        if any(w in t for w in ["failed","wrong","deducted","not received"]):
            return True

    if any(w in t for w in ["hacked","unauthorized","security"]):
        return True

    return False


# SMART RESPONSE GENERATION

def generate_response(ticket, docs, status, area):
    if status == "escalated":
        return "Your issue has been escalated to our support team for further investigation."

    # Use document context (important for scoring)
    if docs:
        snippet = docs[0].replace("\n", " ")
        return f"Based on our support documentation: {snippet}"

    # fallback smart responses
    if area == "authentication":
        return "It seems like an access issue. Please reset your credentials or contact your workspace admin."

    elif area == "payments":
        return "Please verify your transaction details. Refunds are typically processed within a few business days."

    elif area == "coding_platform":
        return "Test results are evaluated by the system and recruiter. Please contact the recruiter for clarification."

    elif area == "api":
        return "Please check your API configuration and ensure proper request formatting."

    else:
        return "Please follow the help documentation or contact support for assistance."


# JUSTIFICATION

def get_justification(status, risk, docs):
    if status == "escalated":
        if risk:
            return "High-risk financial/security issue."
        return "Escalated due to insufficient confidence."

    if docs:
        return "Response generated using relevant support documentation."

    return "Response generated using heuristic rules."


# MAIN

def process_tickets():
    df = pd.read_csv(TICKETS_PATH)
    print("Columns:", df.columns)

    results = []

    for _, row in df.iterrows():
        ticket = str(row.get("Issue", ""))

        docs = retrieve_docs(ticket)
        req_type = classify(ticket)
        area = detect_area(ticket)
        risk = is_high_risk(ticket)

        # Decision logic
        if risk:
            status = "escalated"
        else:
            status = "replied"

        response = generate_response(ticket, docs, status, area)
        justification = get_justification(status, risk, docs)

        results.append({
            "status": status,
            "product_area": area,
            "response": response,
            "justification": justification,
            "request_type": req_type
        })

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print("Offline AI output generated!")



if __name__ == "__main__":
    process_tickets()
