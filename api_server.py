from flask import Flask, request, jsonify
from flask_cors import CORS
from neuro_symbolic_rag import NeuroSymbolicRAG

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize the RAG system
rag = NeuroSymbolicRAG(use_ollama=False)

@app.route('/solve', methods=['POST'])
def solve():
    """API endpoint to solve mathematical queries."""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        result = rag.solve(query, use_simple_parser=True)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Neuro-Symbolic RAG API is running'})

if __name__ == '__main__':
    print("🚀 Starting Neuro-Symbolic Memory Palace API...")
    print("📍 Server running at: http://localhost:5000")
    print("🌐 Open gui.html in your browser to use the interface")
    app.run(debug=True, port=5000)
