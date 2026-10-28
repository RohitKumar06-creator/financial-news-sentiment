
import pandas as pd
import torch
from functools import lru_cache
from transformers import pipeline

MODEL_NAME = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def load_classifier():
    """Load FinBERT once and reuse it."""
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device
    )


def clean_text(text):
    """Normalize and validate input text."""
    if pd.isna(text):
        return ""

    return " ".join(str(text).split())


def analyze_headlines(headlines, classifier=None):
    """Classify headlines and calculate directional sentiment."""

    if classifier is None:
        classifier = load_classifier()

    records = []

    for original in headlines:
        cleaned = clean_text(original)

        if not cleaned:
            continue

        records.append({
            "headline": str(original),
            "cleaned": cleaned
        })

    columns = [
        "headline", "sentiment", "confidence",
        "positive", "negative", "neutral",
        "sentiment_score", "risk_flags", "risk_level"
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    predictions = classifier(
        [r["cleaned"] for r in records],
        top_k=None,
        truncation=True,
        batch_size=8
    )

    output = []

    for record, prediction in zip(records, predictions):
        probs = {
            item["label"].lower(): float(item["score"])
            for item in prediction
        }

        positive = probs["positive"]
        negative = probs["negative"]
        neutral = probs["neutral"]

        label = max(probs, key=probs.get)

        display_label = {
            "positive": "Bullish",
            "negative": "Bearish",
            "neutral": "Neutral"
        }[label]

        text = record["cleaned"].lower()

        # Preliminary risk indicators based on headline wording.
        risk_rules = {
            "Earnings deterioration": [
                "loss", "declining revenue", "earnings miss",
                "weaker earnings", "profit warning"
            ],
            "Debt or liquidity pressure": [
                "debt default", "liquidity crisis",
                "loan defaults", "non-performing loans",
                "debt restructuring"
            ],
            "Regulatory or legal concern": [
                "regulatory investigation", "fraud",
                "accounting issues", "lawsuit", "penalty"
            ],
            "Operational uncertainty": [
                "suspend operations", "factory closure",
                "production halted", "outlook withdrawn",
                "market uncertainty"
            ]
        }

        flags = [
            category
            for category, terms in risk_rules.items()
            if any(term in text for term in terms)
        ]

        # This is a rule-based flag count, not a trained risk model.
        if len(flags) >= 2:
            risk_level = "High"
        elif len(flags) == 1:
            risk_level = "Review"
        else:
            risk_level = "No keyword flag"

        output.append({
            "headline": record["headline"],
            "sentiment": display_label,
            "confidence": probs[label],
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "sentiment_score": positive - negative,
            "risk_flags": ", ".join(flags) if flags else "None",
            "risk_level": risk_level
        })

    return pd.DataFrame(output)


def summarize_sentiment(results):
    """Calculate aggregate sentiment statistics."""

    if results.empty:
        return {
            "total": 0,
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
            "net_sentiment": 0.0
        }

    counts = results["sentiment"].value_counts()

    return {
        "total": len(results),
        "bullish": int(counts.get("Bullish", 0)),
        "bearish": int(counts.get("Bearish", 0)),
        "neutral": int(counts.get("Neutral", 0)),
        "net_sentiment": float(
            results["sentiment_score"].mean()
        )
    }
