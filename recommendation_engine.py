# Simplified recommendation engine for the current project implementation

class RecommendationEngine:
    """
    Generates hiring recommendations based on candidate assessment data.
    """
    
    def __init__(self):
        # Initialize with minimal required components
        self.threshold_strong_hire = 0.8
        self.threshold_hire = 0.65
        self.threshold_consider = 0.5
        
    def generate_recommendation(self, candidate_data):
        """
        Generate a hiring recommendation based on candidate data.
        
        Args:
            candidate_data (dict): Contains resume_score and technical_score
            
        Returns:
            dict: Recommendation, confidence score, and feedback
        """
        # Extract scores
        resume_score = candidate_data.get("resume_score", 0.0)
        
        # Get technical score (overall or calculate if not provided)
        if "technical_score" in candidate_data:
            if isinstance(candidate_data["technical_score"], dict) and "overall" in candidate_data["technical_score"]:
                technical_score = candidate_data["technical_score"]["overall"]
            else:
                technical_score = 0.5  # Default if structure is unexpected
        else:
            technical_score = 0.0
        
        # Calculate weighted score
        # Resume: 40%, Technical: 60%
        weighted_score = (resume_score * 0.4) + (technical_score * 0.6)
        
        # Determine recommendation
        if weighted_score > self.threshold_strong_hire:
            recommendation = "Strong Hire"
            confidence = 0.9
            feedback = "Excellent candidate with strong resume and technical skills."
        elif weighted_score > self.threshold_hire:
            recommendation = "Hire"
            confidence = 0.8
            feedback = "Good candidate with solid qualifications."
        elif weighted_score > self.threshold_consider:
            recommendation = "Consider"
            confidence = 0.6
            feedback = "Potential candidate but has some areas for improvement."
        else:
            recommendation = "Do Not Hire"
            confidence = 0.7
            feedback = "Not a good match for this position."
        
        # Generate strengths and areas for improvement
        strengths = self._generate_strengths(candidate_data)
        improvements = self._generate_improvements(candidate_data)
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "score": weighted_score,
            "feedback": feedback,
            "strengths": strengths,
            "improvements": improvements
        }
    
    def _generate_strengths(self, candidate_data):
        """Generate list of candidate strengths based on scores"""
        strengths = []
        
        if candidate_data.get("resume_score", 0) > 0.7:
            strengths.append("Strong resume with relevant skills and experience")
        
        if "technical_score" in candidate_data:
            tech_score = candidate_data["technical_score"]
            if isinstance(tech_score, dict):
                if tech_score.get("overall", 0) > 0.7:
                    strengths.append("Excellent technical skills")
                if tech_score.get("similarity", 0) > 0.8:
                    strengths.append("Solution closely matches expected approach")
                
        return strengths
    
    def _generate_improvements(self, candidate_data):
        """Generate list of areas for improvement based on scores"""
        improvements = []
        
        if candidate_data.get("resume_score", 0) < 0.6:
            improvements.append("Resume could better highlight relevant skills")
        
        if "technical_score" in candidate_data:
            tech_score = candidate_data["technical_score"]
            if isinstance(tech_score, dict):
                if tech_score.get("overall", 0) < 0.6:
                    improvements.append("Technical skills need improvement")
                if tech_score.get("similarity", 0) < 0.5:
                    improvements.append("Solution approach differs from best practices")
                
        return improvements 