from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
# Enable CORS so our Netlify site can talk to this API
CORS(app)

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
        
        # TODO: Add actual russian-g2p logic here later!
        # For now, we just return a fake test response to make sure the connection works.
        test_response = f"✨ [API Connected!] You sent: {text[:20]}..."
        
        return jsonify({"ipa": test_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Render requires us to bind to 0.0.0.0 and use their provided PORT
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
