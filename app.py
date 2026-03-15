from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import epitran

app = Flask(__name__)
# Enable CORS so our Netlify site can talk to this API
CORS(app)

# Initialize the IPA generator for Russian (Cyrillic)
# We do this globally so it only loads into memory once when the server starts
print("Loading Russian IPA dictionary...")
try:
    epi = epitran.Epitran('rus-Cyrl')
    print("Dictionary loaded!")
except Exception as e:
    print(f"Error loading epitran: {e}")
    epi = None

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
        
        if not epi:
            return jsonify({"error": "IPA dictionary failed to load on the server."}), 500

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

if __name__ == '__main__':
    # Render requires us to bind to 0.0.0.0 and use their provided PORT
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
