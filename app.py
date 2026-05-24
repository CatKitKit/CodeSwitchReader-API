from flask import Flask, request, jsonify
from flask_cors import CORS
import os
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
# Enable CORS so our Netlify site can talk to this API
CORS(app)

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
                ipa_rows.append("") # Keep blank lines blank

        final_ipa_text = '\n'.join(ipa_rows)

        return jsonify({"ipa": final_ipa_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ai-proxy', methods=['POST'])
def ai_proxy():
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return jsonify({"error": "API key not configured on server"}), 500
        
        payload = request.get_json()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={api_key}"
        
        response = requests.post(url, json=payload)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ai-context', methods=['POST'])
def ai_context():
    try:
        data = request.get_json()
        word = data.get('word')
        context = data.get('context')
        
        # 1. Grab the secret key from the cloud environment (not visible to users!)
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return jsonify({"error": "API key not configured on server"}), 500

        # 2. Build the prompt for the AI
        prompt = f"""You are a helpful language teacher. 
Explain this word: '{word}'
Context sentence: '{context}'
Return ONLY a strict JSON object with a 'titleTrans' (literal translation), 'explain' (grammar breakdown in HTML), and 'examples' (2 example sentences in HTML)."""

        # 3. Send the request directly to Google's API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        
        response = requests.post(url, json=payload)
        response_data = response.json()
        
        # 4. Extract the AI's answer and send it back to the web app
        ai_text = response_data['candidates'][0]['content']['parts'][0]['text']
        
        return ai_text # This is the JSON string the web app expects

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Render requires us to bind to 0.0.0.0 and use their provided PORT
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
