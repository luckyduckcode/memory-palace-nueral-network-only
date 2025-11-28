import requests
import json
import os

class SLLMWrapper:
    """
    Wrapper for the Small Language Model (sLLM) running locally via Ollama.
    Acts as the 'Front End' interpreter and 'Back End' explainer.
    """
    def __init__(self, model_name="tinyllama", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def _call_api(self, prompt, format_json=False):
        """Helper to call Ollama API."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if format_json:
            payload["format"] = "json"

        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return None

    def get_embedding(self, text):
        """
        Generates a high-dimensional vector (P-Vec) for the input text.
        Uses Ollama's /api/embeddings endpoint.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get('embedding', [])
        except requests.exceptions.RequestException as e:
            print(f"Error getting embedding: {e}")
            return []

    def parse_intent(self, user_query):
        """
        Stage A: Translates natural language query into a structured Problem Vector.
        
        Returns:
            dict: Structured JSON containing 'intent', 'domain', 'parameters'.
        """
        system_prompt = (
            "You are a mathematical intent parser. "
            "Analyze the user's query and extract the mathematical intent.\n"
            "Output JSON with keys: 'intent' (e.g., 'differentiation', 'integration', 'solve_equation'), "
            "'domain' (e.g., 'calculus', 'algebra'), and 'parameters' (the specific function or equation)."
        )
        
        full_prompt = f"{system_prompt}\n\nUser Query: {user_query}\n\nJSON Output:"
        
        response_text = self._call_api(full_prompt, format_json=True)
        
        if response_text:
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from sLLM: {response_text}")
                return None
        return None

    def explain_solution(self, history_steps, final_result):
        """
        Stage D: Generates a human-readable explanation from the symbolic execution history.
        """
        prompt = (
            f"Explain the following mathematical solution step-by-step to a student.\n"
            f"Steps Taken: {history_steps}\n"
            f"Final Result: {final_result}\n"
            f"Explanation:"
        )
        return self._call_api(prompt)

if __name__ == "__main__":
    # Simple test
    sllm = SLLMWrapper()
    print("Testing connection...")
    
    # Test Generation
    res = sllm.parse_intent("Find the derivative of x^2")
    print(f"Parsed Intent: {res}")
    
    # Test Embedding
    emb = sllm.get_embedding("derivative of x^2")
    print(f"Embedding Length: {len(emb)}")

