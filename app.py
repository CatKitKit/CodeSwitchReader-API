from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import hmac
import time
import threading
try:
    import epitran
    EPITRAN_AVAILABLE = True
except ImportError:
    print("WARNING: epitran is not installed. Russian/Arabic IPA will not work locally.")
    EPITRAN_AVAILABLE = False
import requests
from dotenv import load_dotenv

# Load environment variables from .env file for local testing
load_dotenv()

app = Flask(__name__)

# CORS: explicit origin list (was wildcard until 2026-07-07).
# - https://localhost      = Android Capacitor WebView (androidScheme https)
# - capacitor://localhost  = iOS Capacitor WebView (future)
# - https://codeswitchreader.org = the old web app (still calls /generate-ipa;
#   its /ai-proxy calls now 401 by design — it has no app key and is abandoned)
CORS(app, origins=[
    "https://localhost",
    "capacitor://localhost",
    "https://codeswitchreader.org",
])

# ---------------------------------------------------------------------------
# /ai-proxy protection (added 2026-07-07 — see memory api-security-spec)
# ---------------------------------------------------------------------------

# Shared app token. Set as a Cloud Run env var (Variables tab), same value as
# AI_PROXY_APP_KEY in mobile.js. If unset, /ai-proxy FAILS CLOSED (503) — this
# makes deploy order safe (env var can land before or after this code).
APP_KEY = os.environ.get('APP_KEY', '')

# Only the payload shapes mobile.js actually sends. Model stays server-side.
ALLOWED_PAYLOAD_KEYS = {"contents", "systemInstruction", "generationConfig"}
MAX_BODY_BYTES = 50 * 1024

# In-memory rate limiting is deliberate: gunicorn runs 1 worker (8 threads
# share this process), so no Redis needed at this scale.
RATE_WINDOW_SECS = 60
RATE_MAX_PER_WINDOW = 30      # per-IP sliding window
DAILY_MAX = 5000              # global circuit breaker (protects the Gemini quota)
_rate_lock = threading.Lock()
_ip_hits = {}                 # ip -> [timestamps within the window]
_daily = {"day": None, "count": 0}


def _client_ip():
    # Cloud Run puts the real client IP first in X-Forwarded-For
    fwd = request.headers.get('X-Forwarded-For', '')
    if fwd:
        return fwd.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def _authorized():
    supplied = request.headers.get('X-App-Key', '')
    return bool(APP_KEY) and hmac.compare_digest(supplied, APP_KEY)


def _rate_limited():
    """Return a (response, status) if over a limit, else None."""
    now = time.time()
    today = time.strftime('%Y-%m-%d')
    ip = _client_ip()
    with _rate_lock:
        if _daily["day"] != today:
            _daily["day"] = today
            _daily["count"] = 0
        if _daily["count"] >= DAILY_MAX:
            return jsonify({"error": "Daily limit reached. Please try again tomorrow."}), 429
        hits = [t for t in _ip_hits.get(ip, []) if now - t < RATE_WINDOW_SECS]
        if len(hits) >= RATE_MAX_PER_WINDOW:
            _ip_hits[ip] = hits
            return jsonify({"error": "Too many requests. Please slow down."}), 429
        hits.append(now)
        _ip_hits[ip] = hits
        _daily["count"] += 1
        # Keep the map bounded (stale IPs with no recent hits get dropped)
        if len(_ip_hits) > 10000:
            cutoff = now - RATE_WINDOW_SECS
            for k in list(_ip_hits.keys()):
                if not any(t > cutoff for t in _ip_hits[k]):
                    del _ip_hits[k]
    return None


# Initialize the IPA generator for Russian and Arabic
# We do this globally so it only loads into memory once when the server starts
print("Loading IPA dictionaries...")
epi_instances = {}
try:
    epi_instances['ru'] = epitran.Epitran('rus-Cyrl')
    epi_instances['ar'] = epitran.Epitran('ara-Arab')
    print("Dictionaries loaded!")
except Exception as e:
    print(f"Error loading epitran: {e}")


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "CodeSwitchReader API is running!"})


@app.route('/generate-ipa', methods=['POST'])
def generate_ipa():
    # Deliberately unauthenticated: no key usage, CPU only; the old web app
    # still needs it (mobile went local for RU/AR IPA on 2026-07-04).
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        lang = data.get('lang', 'ru')  # Default to Russian for backward compatibility

        epi = epi_instances.get(lang)
        if not epi:
            return jsonify({"error": f"IPA dictionary for language '{lang}' failed to load or is not supported."}), 500

        # We split the text back into rows, translate each row, and rejoin
        # This ensures the Grid Editor's line-by-line formatting stays intact
        rows = text.split('\n')
        ipa_rows = []

        for row in rows:
            if row.strip():
                # Transliterate generates the IPA symbols
                ipa_row = epi.transliterate(row)
                ipa_rows.append(ipa_row)
            else:
                ipa_rows.append("")  # Keep blank lines blank

        final_ipa_text = '\n'.join(ipa_rows)

        return jsonify({"ipa": final_ipa_text})

    except Exception:
        app.logger.exception("generate-ipa failure")
        return jsonify({"error": "IPA generation failed"}), 500


@app.route('/ai-proxy', methods=['POST'])
def ai_proxy():
    # Order matters: config check -> auth -> rate limit -> size -> shape -> forward.
    # (Auth before rate limit so anonymous floods can't drain the daily budget.)
    if not APP_KEY:
        return jsonify({"error": "Server not configured"}), 503
    if not _authorized():
        return jsonify({"error": "Unauthorized"}), 401
    limited = _rate_limited()
    if limited:
        return limited
    if (request.content_length or 0) > MAX_BODY_BYTES:
        return jsonify({"error": "Request too large"}), 413

    payload = request.get_json(silent=True)
    if (not isinstance(payload, dict)
            or 'contents' not in payload
            or not set(payload).issubset(ALLOWED_PAYLOAD_KEYS)):
        return jsonify({"error": "Bad request"}), 400

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return jsonify({"error": "Server not configured"}), 503

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"
    try:
        response = requests.post(url, json=payload, timeout=30)
        # Pass Gemini's JSON through with a 200 like before — mobile.js checks
        # data.candidates and has its own error UX; don't change the contract.
        return jsonify(response.json())
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 502
    except Exception:
        app.logger.exception("ai-proxy upstream failure")
        return jsonify({"error": "Upstream error"}), 502


# /ai-context was DELETED 2026-07-07: confirmed dead route (zero references in
# mobile.js AND the old web app) that exposed the key with no auth.


if __name__ == '__main__':
    # Render requires us to bind to 0.0.0.0 and use their provided PORT
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
