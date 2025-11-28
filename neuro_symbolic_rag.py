import json
import torch
import spacy
import sympy as sp
from transformers import pipeline
from sllm_wrapper import SLLMWrapper
from mnemonic_model import MPNN, DIM_Net
from symbolic_solver import AdvancedSymbolicSolver

class NeuroSymbolicRAG:
    """
    The main pipeline orchestrating the 4-stage Neuro-Symbolic architecture.
    """
    def __init__(self, use_ollama=True):
        print("Initializing Neuro-Symbolic RAG System...")
        
        # Stage A & D
        self.use_ollama = use_ollama
        if use_ollama:
            self.sllm = SLLMWrapper()
        else:
            self.sllm = None
        
        # Stage B
        self.dim_net = DIM_Net()
        self.mpnn = MPNN(dim_net=self.dim_net)
        
        # Stage C
        self.solver = AdvancedSymbolicSolver()
        
        print("System Initialized.")

    def parse_word_problem(self, problem_text):
        """Parse a word problem using spaCy and a small AI model for better filtering"""
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return {"error": "spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm"}
        
        # Use small AI model for operation classification
        try:
            classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)  # CPU
            candidate_labels = ["addition", "subtraction", "multiplication", "division", "unknown"]
            classification = classifier(problem_text, candidate_labels, truncation=True)
            ai_operation = classification['labels'][0]  # Most likely operation
        except Exception as e:
            ai_operation = "unknown"  # Fallback if model fails
        
        doc = nlp(problem_text)
        entities = {}
        relationships = set()  # Use set to avoid duplicates
        
        # Word to number mapping
        word_nums = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20
        }
        
        # Extract numbers and potential variables
        numbers = []
        for token in doc:
            if token.pos_ == "NUM":
                text_lower = token.text.lower()
                if text_lower in word_nums:
                    num = word_nums[text_lower]
                else:
                    try:
                        num = int(token.text) if token.text.isdigit() else float(token.text)
                    except ValueError:
                        continue
                numbers.append(num)
                entities[token.text] = sp.Symbol(f"x{token.text}")
            elif token.pos_ == "NOUN" and token.dep_ in ["nsubj", "dobj", "pobj"]:
                entities[token.lemma_] = sp.Symbol(token.lemma_)
        
        # Enhanced relationship extraction with AI input
        rule_based_relations = set()
        ai_relations = set()
        
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in ["twice", "double", "each", "per", "every"]:
                rule_based_relations.add("multiplication")
            elif lemma in ["sum", "total", "add", "plus", "give", "gave", "gives"]:
                rule_based_relations.add("addition")
            elif lemma in ["difference", "subtract", "minus", "move", "moved", "transfer", "shift", "take", "took", "remain", "remaining", "left"]:
                rule_based_relations.add("subtraction")
            elif lemma in ["product", "multiply", "times"]:
                rule_based_relations.add("multiplication")
            elif lemma in ["quotient", "divide", "divided"]:
                rule_based_relations.add("division")
        
        # AI as fallback, not override
        if ai_operation != "unknown" and not rule_based_relations:
            ai_relations.add(ai_operation)
        
        relationships = list(rule_based_relations | ai_relations)
        
        # Direct solving for common patterns based on relationships
        solution = None
        equation = None
        if "subtraction" in relationships and len(numbers) >= 2:
            unique_nums = sorted(set(numbers))
            if len(unique_nums) >= 2:
                solution = unique_nums[-1] - unique_nums[0]  # max - min
                equation = f"{unique_nums[-1]} - {unique_nums[0]} = {solution}"
        elif "addition" in relationships and len(numbers) >= 2:
            solution = sum(numbers)
            equation = " + ".join(map(str, numbers)) + f" = {solution}"
        elif "multiplication" in relationships and len(numbers) >= 2:
            solution = 1
            for n in numbers:
                solution *= n
            equation = " * ".join(map(str, numbers)) + f" = {solution}"
        elif "division" in relationships and len(numbers) >= 2 and numbers[1] != 0:
            solution = numbers[0] / numbers[1]
            equation = f"{numbers[0]} / {numbers[1]} = {solution}"
        
        return {
            "entities": {k: str(v) for k, v in entities.items()},
            "relationships": relationships,
            "equation": equation,
            "solution": solution,
            "ai_operation": ai_operation  # Include AI-detected operation
        }

    def solve(self, user_query, use_simple_parser=True):
        """
        Solves a math problem using the Memory Palace RAG.
        """
        print(f"\n--- Processing Query: '{user_query}' ---")
        
        # --- Stage A: Intent Parsing & Vectorization ---
        print("[Stage A] Parsing Intent...")
        
        if use_simple_parser or not self.use_ollama:
            # Simple rule-based parser for PoC
            intent_data = self._simple_parse(user_query)
        else:
            intent_data = self.sllm.parse_intent(user_query)
            
        if not intent_data:
            return {"error": "Could not parse intent."}
        
        # Check if word problem with direct solution
        if 'parsed_data' in intent_data and intent_data['parsed_data'].get('solution') is not None:
            result = intent_data['parsed_data']['solution']
            explanation = self._generate_explanation([], result, intent_data)
            # Mock tier2_location
            tier2_location = f"word_problem_{intent_data.get('domain', 'math')}"
            return {
                "query": user_query,
                "result": str(result),
                "tier2_location": tier2_location,
                "path": ["word_problem_parsing"],
                "explanation": explanation
            }
        
        print(f"  Intent: {intent_data.get('intent')}")
        print(f"  Domain: {intent_data.get('domain')}")
        print(f"  Expression: {intent_data.get('expression')}")
        
        expression = intent_data.get('expression')
        
        # --- Stage B: Memory Palace Search ---
        print("[Stage B] Searching Memory Palace...")
        
        # Map intent to goal concept
        intent = intent_data.get('intent')
        goal_concept = self._intent_to_concept(intent)
        coords = self.mpnn.concept_locations.get(goal_concept, (1,1,1))
        
        print(f"  Goal Loci: {goal_concept} at {coords}")
        
        # Path Derivation
        start_concept = intent_data.get('domain', 'calculus')
        if start_concept not in self.mpnn.knowledge_graph:
            start_concept = 'calculus'
            
        print(f"  Deriving Path from '{start_concept}' to '{goal_concept}'...")
        path_sequence = self.mpnn.derive_path(start_concept, goal_concept)
        print(f"  Path Sequence: {path_sequence}")
        
        # --- Stage C: Symbolic Execution ---
        print("[Stage C] Executing Symbolic Path...")
        final_result, history = self.solver.execute_path(expression, path_sequence)
        print(f"  Symbolic Result: {final_result}")
        
        # --- Stage D: Explanation Generation ---
        print("[Stage D] Generating Explanation...")
        explanation = self._generate_explanation(history, final_result, intent_data)
        
        # --- Consolidation: Store in Tier 2 Memory Palace ---
        print("[Consolidation] Storing result in Tier 2 Memory Palace...")
        storage_key, tier2_coords = self.mpnn.consolidate_memory(goal_concept, explanation, path_sequence)
        print(f"  Stored at Tier 2 Loci: {storage_key} (linked to {goal_concept})")
        
        return {
            "query": user_query,
            "intent": intent_data,
            "path": path_sequence,
            "result": final_result,
            "explanation": explanation,
            "tier2_location": storage_key
        }
    
    def _simple_parse(self, query):
        """Enhanced rule-based parser for comprehensive math queries."""
        query_lower = query.lower()
        
        # Check if it's a word problem
        import re
        has_numbers = bool(re.search(r'\d+', query))
        word_problem_keywords = ['has', 'gives', 'takes', 'how many', 'how much', 'left', 'remain', 'total', 'sum', 'difference']
        is_word_problem = has_numbers and any(kw in query_lower for kw in word_problem_keywords)
        
        if is_word_problem:
            parsed = self.parse_word_problem(query)
            if 'error' not in parsed:
                # Set intent based on relationships
                if 'addition' in parsed['relationships']:
                    intent = 'addition'
                    domain = 'arithmetic'
                elif 'subtraction' in parsed['relationships']:
                    intent = 'subtraction'
                    domain = 'arithmetic'
                elif 'multiplication' in parsed['relationships']:
                    intent = 'multiplication'
                    domain = 'arithmetic'
                elif 'division' in parsed['relationships']:
                    intent = 'division'
                    domain = 'arithmetic'
                else:
                    intent = 'word_problem'
                    domain = 'arithmetic'
                
                expression = parsed.get('equation', 'x')
                return {
                    'intent': intent,
                    'domain': domain,
                    'expression': expression,
                    'parsed_data': parsed
                }
        
        # Detect intent (existing logic)
        if 'derivative' in query_lower or 'differentiate' in query_lower or 'd/dx' in query_lower:
            intent = 'differentiation'
            domain = 'calculus'
        elif 'integrate' in query_lower or 'integral' in query_lower or '∫' in query_lower:
            intent = 'integration'
            domain = 'calculus'
        elif 'solve' in query_lower and ('equation' in query_lower or '=' in query):
            intent = 'solve_equation'
            domain = 'algebra'
        elif 'simplify' in query_lower:
            intent = 'simplify'
            domain = 'algebra'
        elif 'expand' in query_lower:
            intent = 'expand'
            domain = 'algebra'
        elif 'factor' in query_lower:
            intent = 'factor'
            domain = 'algebra'
        elif 'limit' in query_lower:
            intent = 'limit'
            domain = 'calculus'
        elif 'taylor' in query_lower or 'series' in query_lower:
            intent = 'series'
            domain = 'calculus'
        elif 'matrix' in query_lower or 'determinant' in query_lower or 'eigenvalue' in query_lower:
            intent = 'matrix_operations'
            domain = 'linear_algebra'
        elif 'create formula' in query_lower or 'define formula' in query_lower:
            intent = 'create_formula'
            domain = 'custom'
        else:
            intent = 'unknown'
            domain = 'math'
        
        # Extract expression (enhanced heuristic)
        expression = None
        if 'of ' in query_lower:
            # Find the position in lowercase, extract from original
            pos = query_lower.find('of ') + 3
            expression = query[pos:].strip()
            expression = expression.rstrip('.?!')
        elif 'solve equation:' in query_lower:
            pos = query_lower.find('solve equation:') + len('solve equation:')
            expression = query[pos:].strip()
            expression = expression.rstrip('.?!')
        elif 'solve ' in query_lower:
            pos = query_lower.find('solve ') + 6
            expression = query[pos:].strip()
            expression = expression.rstrip('.?!')
        elif ':' in query:  # For "simplify: (x+1)^2"
            expression = query.split(':')[-1].strip()
            expression = expression.rstrip('.?!')
        
        # Convert ^ to ** for Python/SymPy
        if expression:
            expression = expression.replace('^', '**')
        
        return {
            'intent': intent,
            'domain': domain,
            'expression': expression or 'x'
        }
    
    def _intent_to_concept(self, intent):
        """Map intent to goal concept."""
        mapping = {
            'differentiation': 'differentiation',
            'integration': 'integration',
            'solve_equation': 'linear_equations',
            'simplify': 'simplify',
            'expand': 'expand',
            'factor': 'factor',
            'limit': 'limits',
            'series': 'taylor_series',
            'matrix_operations': 'matrix_operations',
            'addition': 'algebra',  # Map arithmetic to algebra
            'subtraction': 'algebra',
            'multiplication': 'algebra',
            'division': 'algebra',
            'word_problem': 'algebra',
            'unknown': 'calculus'
        }
        return mapping.get(intent, 'calculus')
    
    def _generate_explanation(self, history, result, intent_data):
        """Generate explanation with branched retrieval for richer context."""
        # Get goal concept for branched retrieval
        intent = intent_data.get('intent')
        goal_concept = self._intent_to_concept(intent)
        
        # Retrieve similar explanations from neighboring locations
        try:
            similar_explanations = self.mpnn.retrieve_similar_explanations(goal_concept, radius=1)
            branched_context = []
            for exp_entry in similar_explanations[-3:]:  # Get last 3 for context
                if isinstance(exp_entry, dict) and 'explanation' in exp_entry:
                    # Extract key insights from similar explanations
                    exp_text = exp_entry['explanation']
                    if len(exp_text) > 100:  # Truncate long explanations
                        exp_text = exp_text[:100] + "..."
                    branched_context.append(exp_text)
        except Exception as e:
            branched_context = []
        
        if self.use_ollama and self.sllm:
            try:
                return self.sllm.explain_solution(history, result, branched_context)
            except:
                pass
        
        # Fallback: Simple template-based explanation with branched context
        expr = intent_data.get('expression')
        
        explanation = f"To solve this {intent} problem:\n"
        if 'parsed_data' in intent_data:
            parsed = intent_data['parsed_data']
            explanation += f"  • Parsed entities: {parsed['entities']}\n"
            explanation += f"  • Relationships: {', '.join(parsed['relationships'])}\n"
            if parsed['equation']:
                explanation += f"  • Equation: {parsed['equation']}\n"
        
        # Add branched context if available
        if branched_context:
            explanation += f"\nRelated concepts and methods:\n"
            for i, context in enumerate(branched_context, 1):
                explanation += f"  • Similar approach {i}: {context}\n"
        
        for step in history:
            explanation += f"  • {step}\n"
        explanation += f"\nFinal Result: {result}"
        
        return explanation

if __name__ == "__main__":
    # Initialize without requiring Ollama
    rag = NeuroSymbolicRAG(use_ollama=False)
    
    # Test queries
    test_queries = [
        "Find the derivative of x^2",
        "Find the derivative of x^3 + 2*x",
        "Integrate 2*x",
    ]
    
    for query in test_queries:
        print("\n" + "="*60)
        result = rag.solve(query, use_simple_parser=True)
        
        if "error" not in result:
            print("\n--- FINAL OUTPUT ---")
            print(f"Query: {result['query']}")
            print(f"Result: {result['result']}")
            print(f"Tier 2 Location: {result['tier2_location']}")
            print(f"\nExplanation:\n{result['explanation']}")
