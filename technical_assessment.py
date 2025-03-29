import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TechnicalAssessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        # Load reference solutions
        self.reference_solutions = {
            "problem1": ["solution1.py", "solution2.py"],
            "problem2": ["solution3.py", "solution4.py"],
            # More problems and solutions
        }
        
    def analyze_code(self, code, problem_id):
        # Parse the code
        try:
            parsed = ast.parse(code)
            
            # Extract metrics
            metrics = {
                "num_functions": len([node for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)]),
                "num_classes": len([node for node in ast.walk(parsed) if isinstance(node, ast.ClassDef)]),
                "num_loops": len([node for node in ast.walk(parsed) 
                                if isinstance(node, (ast.For, ast.While))]),
                "num_conditionals": len([node for node in ast.walk(parsed) 
                                       if isinstance(node, ast.If)]),
                "complexity": self._calculate_complexity(parsed)
            }
            
            # Compare with reference solutions
            similarity = self._compare_with_references(code, problem_id)
            
            return {
                "metrics": metrics,
                "similarity": similarity,
                "syntax_error": False
            }
            
        except SyntaxError:
            return {
                "metrics": {},
                "similarity": 0.0,
                "syntax_error": True
            }
    
    def _calculate_complexity(self, parsed):
        # Simple cyclomatic complexity calculation
        complexity = 1  # Base complexity
        for node in ast.walk(parsed):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
        return complexity
    
    def _compare_with_references(self, code, problem_id):
        if problem_id not in self.reference_solutions:
            return 0.0
            
        reference_codes = []
        for solution_file in self.reference_solutions[problem_id]:
            with open(solution_file, 'r') as f:
                reference_codes.append(f.read())
                
        # Preprocess code (remove comments, whitespace)
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\s+', ' ', code).strip()
        
        reference_codes = [re.sub(r'#.*', '', ref) for ref in reference_codes]
        reference_codes = [re.sub(r'\s+', ' ', ref).strip() for ref in reference_codes]
        
        # Calculate similarity
        all_codes = [code] + reference_codes
        tfidf_matrix = self.vectorizer.fit_transform(all_codes)
        
        # Get the highest similarity with any reference solution
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        return float(max(similarities)) if len(similarities) > 0 else 0.0 