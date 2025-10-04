import os
import re
import numpy as np
import feedparser
from textblob import TextBlob
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException , Response
from pydantic import BaseModel
import google.generativeai as genai


app = FastAPI(title="Trend Tag Analyzer")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")
genai.configure(api_key=API_KEY)

def extract_urls(text, max_urls=5):
    urls = re.findall(r"(https?://[^\s\)\]\`]+)", text)
    cleaned = []
    for u in urls:
        if u.count("http") > 1:
            u = "http" + u.split("http")[-1]
        u = u.strip(" ,)`(")
        cleaned.append(u)
    return list(dict.fromkeys(cleaned))[:max_urls]

def gemini_generate_new_tags(base_tag, max_tags=5):
    try:
        prompt = f"""
        Generate {max_tags} new creative, visual, or cultural trend tags related to '{base_tag}'.
        Focus on aesthetics, artistic movements, or online cultural styles.
        Return only a comma-separated list of tags, no explanations.
        """
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
        tags = [t.strip() for t in re.split(r"[,\n]", text) if t.strip()]
        return tags[:max_tags]
    except Exception as e:
        print(f"Gemini tag generation error: {e}")
        return []

def gemini_fetch_urls(tag, max_urls=5):
    try:
        prompt = f"""
        Give me {max_urls} popular and reliable RSS feed URLs or websites about the visual or artistic style '{tag}'.
        Focus on current trends, design movements, and aesthetic culture updates.
        """
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return extract_urls(response.text, max_urls=max_urls)
    except Exception as e:
        print(f"Gemini URL fetch error: {e}")
        return []

def fetch_rss_articles(rss_url, days_back=30):
    feed = feedparser.parse(rss_url)
    articles = []
    cutoff = datetime.now() - timedelta(days=days_back)
    for entry in feed.entries:
        pub_date = getattr(entry, "published_parsed", None)
        if pub_date:
            pub_date = datetime(*pub_date[:6])
        else:
            pub_date = datetime.now()
        if pub_date >= cutoff:
            content = getattr(entry, "summary", "") or getattr(entry, "description", "")
            articles.append({
                "title": entry.title,
                "content": content,
                "published": pub_date,
                "length": len(content.split())
            })
    return articles

def analyze_sentiment(articles):
    polarities, subjectivities = [], []
    for art in articles:
        text = art["title"] + ". " + art["content"]
        analysis = TextBlob(text).sentiment
        polarities.append(analysis.polarity)
        subjectivities.append(analysis.subjectivity)
    if not polarities:
        return 0.5, 0.5, 0.5
    polarity_mean = float((np.mean(polarities) + 1) / 2)
    polarity_std = float(np.std(polarities))
    subjectivity_mean = float(np.mean(subjectivities))
    return polarity_mean, polarity_std, subjectivity_mean

def compute_momentum_volatility(articles, days=7):
    if not articles:
        return 0.5, 0.2
    today = datetime.now().date()
    counts = np.zeros(days)
    for art in articles:
        delta = (today - art["published"].date()).days
        if delta < days:
            counts[days - delta - 1] += 1
    momentum = (counts[-1] - counts[0]) / max(1, np.mean(counts))
    volatility = np.std(counts) / (np.mean(counts) + 1)
    return float(np.clip(0.5 + momentum, 0, 1)), float(np.clip(volatility, 0, 1))

def compute_content_depth_recency(articles):
    if not articles:
        return 0.5, 0.5
    lengths = [a["length"] for a in articles]
    depth = float(np.mean(lengths) / max(1, max(lengths)))
    recency = float(np.mean([1 / (1 + (datetime.now() - a["published"]).days) for a in articles]))
    return depth, recency

def analyze_tag(tag):
    urls_found = gemini_fetch_urls(tag, max_urls=5)
    all_articles = []
    for url in urls_found:
        all_articles.extend(fetch_rss_articles(url))
    social_volume = len(all_articles)
    polarity_mean, polarity_std, subjectivity_mean = analyze_sentiment(all_articles)
    momentum, volatility = compute_momentum_volatility(all_articles)
    depth, recency = compute_content_depth_recency(all_articles)
    social_score = float(np.log1p(social_volume) * 0.3)
    sentiment_score = float(0.4 * polarity_mean + 0.1 * (1 - polarity_std))
    momentum_score = float(momentum * 0.1)
    volatility_penalty = float((1 - volatility) * 0.1)
    content_depth_score = float(depth * 0.05)
    recency_score = float(recency * 0.05)
    reception_score = (
        social_score
        + sentiment_score
        + momentum_score
        + volatility_penalty
        + content_depth_score
        + recency_score
    )
    intensity = reception_score
    if reception_score > 0.65:
        judgement = "Strong Potential"
    elif reception_score < 0.35:
        judgement = "Weak Potential"
    else:
        judgement = "Moderate Potential"
    return {
        "tag": tag,
        "urls_found": urls_found,
        "num_articles": social_volume,
        "social_score": social_score,
        "sentiment_score": sentiment_score,
        "momentum_score": momentum_score,
        "volatility_penalty": volatility_penalty,
        "content_depth_score": content_depth_score,
        "recency_score": recency_score,
        "reception_score": float(reception_score),
        "intensity": float(intensity),
        "judgement": judgement
    }

def analyze_tags(tags):
    details = [analyze_tag(tag) for tag in tags]
    reception = float(np.mean([d["reception_score"] for d in details]))
    intensity = reception
    return {
        "reception": reception,
        "intensity": float(intensity),
        "details": details
    }

class AnalyzeRequest(BaseModel):
    base_tag: str
    max_new_tags: int = 5

@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    try:
        new_tags = gemini_generate_new_tags(req.base_tag, req.max_new_tags)
        all_tags = [req.base_tag] + new_tags
        analysis = analyze_tags(all_tags)
        return {
            "base_tag": req.base_tag,
            "generated_tags": new_tags,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.head("/ping")
@app.get("/ping")
def ping():
    return Response(status_code=200)

@app.get("/")
def root():
    return {"message": "Trend Tag Analyzer API is running ✨"}
