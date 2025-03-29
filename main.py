from resume_screening import ResumeScreener
from technical_assessment import TechnicalAssessor
from recommendation_engine import RecommendationEngine

class AutomatedInterviewSystem:
    def __init__(self):
        self.resume_screener = ResumeScreener()
        self.technical_assessor = TechnicalAssessor()
        self.recommendation_engine = RecommendationEngine()
        
    def process_candidate(self, resume_path, code_submission, job_id):
        # Step 1: Resume screening
        with open(resume_path, 'r') as f:
            resume_text = f.read()
        resume_score = self.resume_screener.predict(resume_text)
        
        # Step 2: Technical assessment
        tech_assessment = self.technical_assessor.analyze_code(code_submission, job_id)
        tech_score = {
            "metrics": tech_assessment["metrics"],
            "similarity": tech_assessment["similarity"],
            "overall": 0.8 if not tech_assessment["syntax_error"] else 0.3
        }
        
        # Step 3: Final recommendation
        candidate_data = {
            "resume_score": resume_score[1],  # Assuming index 1 is the positive class
            "technical_score": tech_score
        }
        
        recommendation = self.recommendation_engine.generate_recommendation(candidate_data)
        
        return {
            "candidate_data": candidate_data,
            "recommendation": recommendation
        }

# Example usage
if __name__ == "__main__":
    system = AutomatedInterviewSystem()
    result = system.process_candidate(
        "candidate_resume.pdf",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "software_engineer_123"
    )
    
    print("Final recommendation:", result["recommendation"]["recommendation"])
    print("Confidence:", result["recommendation"]["confidence"])
    print("Feedback:", result["recommendation"]["feedback"]) 