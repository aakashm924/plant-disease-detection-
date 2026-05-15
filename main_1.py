"""
Enhanced Plant Chatbot Backend — main.py
=========================================
Improvements over original:
 1. Uses Claude claude-sonnet-4-20250514 via Anthropic API for truly intelligent answers
 2. Keeps TF-IDF FAQ as a fast-path for common questions (< 0.4s response)
 3. Maintains conversation history per session (last 6 turns)
 4. Returns structured JSON with source + confidence
 5. /suggest endpoint: recommends follow-up questions based on context
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import anthropic
import json
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent

# ── FAQ setup ────────────────────────────────────────────────────────────────
faq_path = BASE_DIR / 'plant_faq.csv'
if faq_path.exists():
    faq_df = pd.read_csv(faq_path)
else:
    faq_df = pd.DataFrame({'question': [], 'answer': []})

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=5000)
faq_vectors = None
if len(faq_df) > 0:
    faq_vectors = vectorizer.fit_transform(faq_df['question'].astype(str))

def retrieve_faq(query: str, threshold: float = 0.50):
    if faq_vectors is None or len(faq_df) == 0:
        return None
    qv = vectorizer.transform([query.lower()])
    sims = cosine_similarity(qv, faq_vectors).flatten()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        return {"answer": faq_df.iloc[best_idx]['answer'], "score": best_score}
    return None

# ── Anthropic client ──────────────────────────────────────────────────────────
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are PlantDoc AI — an expert botanist, plant pathologist, and agronomist.
Your job is to help gardeners, farmers, and plant enthusiasts with:
- Plant disease identification and treatment
- Organic and chemical treatment options
- Watering, fertilisation, and soil care
- Pest control and prevention
- Seasonal plant care tips

Rules:
- Always be specific, practical, and concise (3–6 sentences max per response)
- Mention whether a treatment is organic or chemical when relevant
- If the question is completely unrelated to plants/agriculture, politely redirect
- Use bullet points for lists (3 items max) — keep them tight
- Never make up chemical names — if unsure, say "consult your local extension service"
- If a disease sounds serious (late blight, citrus greening), flag the urgency clearly"""

# Simple in-memory session history (keyed by session_id passed from frontend)
_histories: Dict[str, List] = {}

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(silent=True) or {}
    user_query = data.get('query', '').strip()
    session_id = data.get('session_id', 'default')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Fast-path: FAQ match
    faq_hit = retrieve_faq(user_query)
    if faq_hit:
        return jsonify({
            'answer': faq_hit['answer'],
            'source': 'FAQ',
            'confidence': round(faq_hit['score'], 2),
        })

    # Build conversation history
    history = _histories.get(session_id, [])
    history.append({"role": "user", "content": user_query})

    # Keep only last 6 messages (3 turns)
    trimmed = history[-6:]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=trimmed,
        )
        answer = response.content[0].text.strip()

        # Store assistant reply in history
        history.append({"role": "assistant", "content": answer})
        _histories[session_id] = history[-8:]  # cap at 8 entries

        return jsonify({
            'answer': answer,
            'source': 'AI',
            'confidence': 'claude-sonnet',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/suggest', methods=['POST'])
def suggest():
    """Return 3 contextual follow-up questions based on last bot response."""
    data = request.get_json(silent=True) or {}
    context = data.get('context', '')
    if not context:
        return jsonify({'suggestions': []})

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": f"Based on this plant disease context, suggest exactly 3 short follow-up questions a user might ask. Return ONLY a JSON array of 3 strings, nothing else.\n\nContext: {context}"
            }]
        )
        raw = resp.content[0].text.strip()
        suggestions = json.loads(raw)
        return jsonify({'suggestions': suggestions[:3]})
    except Exception:
        return jsonify({'suggestions': [
            "What are organic treatment options?",
            "How do I prevent this from spreading?",
            "When should I apply fungicide?"
        ]})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'chatbot': 'claude-sonnet-4-20250514'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False, use_reloader=False)
