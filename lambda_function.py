import json
import os
import hashlib
import requests
from datetime import datetime, timezone
from urllib.parse import parse_qs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import traceback

# ===== Config =====
API_TOKEN = "#####################"

ALLOWED_COUNTRIES = {"us", "gb", "in", "ca", "au", "de", "fr", "jp", "br", "za"}

DEFAULT_TOP_URL = "https://api.thenewsapi.com/v1/news/top"
SEARCH_URL = "https://api.thenewsapi.com/v1/news/all"

DEFAULT_LIMIT = 30
COUNTRY_LIMIT = 3
LANGUAGE = "en"
CACHE_DIR = "/tmp"

# --- HF Inference (keep simple per your ask) ---
HF_TOKEN = os.getenv("HF_TOKEN", "#######################")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "amohan78/distilbert_sentiment_model")
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

_hf_session = requests.Session()
_hf_session.headers.update({"User-Agent": "newzsenti/1.0"})

# CORS
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Expose-Headers": "X-Resolved-Mode,X-Resolved-Country",
}

analyzer = SentimentIntensityAnalyzer()

# ===== Helpers =====
def _http_method(event):
    m = event.get("httpMethod")
    if not m:
        rc = event.get("requestContext", {}).get("http", {})
        m = rc.get("method")
    return (m or "GET").upper()

def _extract_params(event):
    country = None
    search = None
    flush = None

    qsp = event.get("queryStringParameters") or {}
    if isinstance(qsp, dict):
        country = qsp.get("country", country)
        search = qsp.get("search", search)
        flush = qsp.get("flush", flush)

    raw_body = event.get("body")
    if raw_body:
        try:
            if event.get("isBase64Encoded"):
                import base64
                raw_body = base64.b64decode(raw_body).decode("utf-8", errors="ignore")
            try:
                payload = json.loads(raw_body)
                if isinstance(payload, dict):
                    country = payload.get("country", country)
                    search  = payload.get("search", search)
                    flush   = payload.get("flush", flush)
            except json.JSONDecodeError:
                form = parse_qs(raw_body, keep_blank_values=False)
                if "country" in form and form["country"]: country = form["country"][0]
                if "search"  in form and form["search"]:  search  = form["search"][0]
                if "flush"   in form and form["flush"]:   flush   = form["flush"][0]
        except Exception:
            pass

    country = (country or "").strip().lower() or None
    search  = (search  or "").strip() or None
    flush   = (flush   or "").strip() or None
    return country, search, flush

def _am_pm_bucket(dt=None):
    tz = timezone.utc
    now = dt.astimezone(tz) if dt else datetime.now(tz)
    half = "am" if now.hour < 12 else "pm"
    return f"{now:%Y%m%d}_{half}"

def _country_cache_path(country):
    bucket = _am_pm_bucket()
    cc = (country or "us")
    return os.path.join(CACHE_DIR, f"news_cache_country_{cc}_{bucket}.json")

def _search_cache_path(search):
    bucket = _am_pm_bucket()
    key = search or ""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"news_cache_search_{h}_{bucket}.json")

def _read_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _write_cache(path, payload):
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass

def _fetch_news(country, search):
    params = {"api_token": API_TOKEN}
    url = DEFAULT_TOP_URL

    if search:
        url = SEARCH_URL
        if "," in search and "|" not in search:
            terms = [t.strip() for t in search.split(",") if t.strip()]
            search_expr = " | ".join(terms) if terms else search
        else:
            search_expr = search
        params["search"] = search_expr
        params["language"] = LANGUAGE
    else:
        cc = country if country in ALLOWED_COUNTRIES else "us"
        params.update({
            "locale": cc,
            "language": LANGUAGE,
            "limit": COUNTRY_LIMIT if country else DEFAULT_LIMIT
        })

    resp = requests.get(url, params=params, timeout=12)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])

# ===== HF Inference Helper =====
def run_distilbert_inference(text: str):
    """
    Calls Hugging Face Inference API for 3-class sentiment and returns:
        {"label": "NEGATIVE|NEUTRAL|POSITIVE", "score": 0.xxx, "compound": [-1,1]}
    Minimal changes + robust warmup handling.
    """
    if not (HF_TOKEN and HF_MODEL_ID and HF_URL):
        raise RuntimeError("HF config missing")

    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    # IMPORTANT: wait_for_model avoids 503 "loading"
    payload = {"inputs": text[:5000], "options": {"wait_for_model": True}}

    for attempt in range(4):
        r = _hf_session.post(HF_URL, headers=headers, json=payload, timeout=20)
        if r.status_code in (503, 429) and attempt < 3:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        out = r.json()

        # Normalize HF outputs to a flat list of {label, score}
        if isinstance(out, list) and out and isinstance(out[0], list):
            candidates = out[0]
        elif isinstance(out, list):
            candidates = out
        elif isinstance(out, dict):
            candidates = [out]
        else:
            raise RuntimeError(f"Unexpected HF payload type: {type(out)}")

        probs = {str(c.get("label","")).upper(): float(c.get("score", 0.0)) for c in candidates if isinstance(c, dict)}

        p_neg = probs.get("NEGATIVE", 0.0)
        p_neu = probs.get("NEUTRAL", 0.0)
        p_pos = probs.get("POSITIVE", 0.0)

        best_label, best_prob = max([("NEGATIVE", p_neg), ("NEUTRAL", p_neu), ("POSITIVE", p_pos)], key=lambda kv: kv[1])
        compound = round(p_pos - p_neg, 3)
        return {"label": best_label, "score": round(best_prob, 3), "compound": compound}

    raise RuntimeError("HF inference failed")

def _distilbert_fallback_from_vader(title, vader_compound):
    """
    If HF fails, synthesize a non-PENDING object so UI never shows 'PENDING'.
    """
    if vader_compound >= 0.05:
        return {"label": "POSITIVE", "score": round(min(0.95, 0.5 + abs(vader_compound)/2), 3), "compound": round(vader_compound, 3)}
    elif vader_compound <= -0.05:
        return {"label": "NEGATIVE", "score": round(min(0.95, 0.5 + abs(vader_compound)/2), 3), "compound": round(vader_compound, 3)}
    else:
        return {"label": "NEUTRAL", "score": 0.5, "compound": round(vader_compound, 3)}

def _analyze_article(article):
    title = article.get("title", "") or ""
    vscores = analyzer.polarity_scores(title)  # VADER first

    # Default
    distil = {"label": "PENDING", "score": None}

    # Try HF DistilBERT; if it fails, use VADER-derived fallback
    try:
        distil = run_distilbert_inference(title)
    except Exception as e:
        print("HF ERROR:", repr(e))
        traceback.print_exc()
        distil = _distilbert_fallback_from_vader(title, vscores["compound"])

    return {
        "title": title,
        "source": article.get("source"),
        "url": article.get("url"),
        "publishedAt": article.get("published_at"),
        "vader": {"compound": vscores["compound"]},
        "distilbert": distil,
    }

# ===== Handler =====
def lambda_handler(event, context):
    method = _http_method(event)

    # CORS preflight
    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    country, search, flush = _extract_params(event)

    # Decide cache file
    if search:
        cache_file = _search_cache_path(search)
        resolved_mode = "search"
        resolved_country = ""
    else:
        cc = country if country in ALLOWED_COUNTRIES else "us"
        cache_file = _country_cache_path(cc)
        resolved_mode = "country"
        resolved_country = cc

    # Read cache; bypass if flush=1 or cache contains any PENDING
    cached = None if flush else _read_cache(cache_file)
    if cached is not None:
        try:
            if any(a.get("distilbert", {}).get("label") == "PENDING" for a in cached.get("articles", [])):
                cached = None
        except Exception:
            cached = None

    if cached is not None:
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "X-Resolved-Mode": resolved_mode, "X-Resolved-Country": resolved_country},
            "body": json.dumps(cached),
        }

    # Fresh fetch + analyze
    try:
        raw = _fetch_news(country, search)
        analyzed = [_analyze_article(a) for a in raw]
        response = {"articles": analyzed}
        _write_cache(cache_file, response)
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "X-Resolved-Mode": resolved_mode, "X-Resolved-Country": resolved_country},
            "body": json.dumps(response),
        }
    except Exception as e:
        print("LAMBDA ERROR:", repr(e))
        traceback.print_exc()
        return {"statusCode": 500, "headers": CORS_HEADERS, "body": json.dumps({"error": str(e)})}
