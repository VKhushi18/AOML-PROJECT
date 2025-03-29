# Simplified resume screening for the current project implementation
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeScreener:
    """
    Screens resumes for job relevance and predicts candidate suitability.
    """
    
    def __init__(self):
        # Initialize with job-specific keywords
        self.job_skills = {
            "software_engineer": [
                "python", "java", "javascript", "c++", "algorithms", "data structures",
                "software development", "git", "agile", "testing", "debugging"
            ],
            "data_scientist": [
                "python", "r", "sql", "machine learning", "statistics", "data analysis",
                "pandas", "numpy", "visualization", "big data", "tensorflow", "pytorch"
            ],
            "product_manager": [
                "product development", "user research", "agile", "roadmap", "stakeholder",
                "market analysis", "user stories", "prioritization", "strategy"
            ]
        }
        
        # Initialize TF-IDF vectorizer for text comparison
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def predict(self, resume_text, job_id="software_engineer"):
        """
        Predict if a resume is suitable for a given job.
        
        Args:
            resume_text (str): The text content of the resume
            job_id (str): The job identifier to match against
            
        Returns:
            tuple: (prediction label, confidence score)
        """
        # Extract relevant skills from resume
        skills_found = self._extract_skills(resume_text, job_id)
        
        # Calculate match score based on skills
        job_skills = self.job_skills.get(job_id, self.job_skills["software_engineer"])
        match_score = len(skills_found) / len(job_skills) if job_skills else 0
        
        # Calculate text similarity with job description
        job_description = self._get_job_description(job_id)
        similarity_score = self._calculate_similarity(resume_text, job_description)
        
        # Combine scores (70% skills match, 30% text similarity)
        final_score = (match_score * 0.7) + (similarity_score * 0.3)
        
        # Make prediction
        prediction = "suitable" if final_score > 0.5 else "not_suitable"
        
        return (prediction, final_score)
    
    def _extract_skills(self, text, job_id):
        """Extract job-relevant skills from resume text"""
        skills_found = set()
        job_skills = self.job_skills.get(job_id, self.job_skills["software_engineer"])
        
        for skill in job_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                skills_found.add(skill)
                
        return skills_found
    
    def _calculate_similarity(self, text1, text2):
        """Calculate cosine similarity between two text documents"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.5  # Default fallback value
    
    def _get_job_description(self, job_id):
        """Get job description for a given job ID"""
        descriptions = {
            "software_engineer": """
                Software Engineer with strong programming skills in Python, Java or C++.
                Experience with algorithms, data structures, and software development practices.
                Knowledge of git, agile methodologies, and testing frameworks.
                Ability to debug complex issues and optimize code performance.
            """,
            "data_scientist": """
                Data Scientist with expertise in Python, R, and SQL.
                Strong background in machine learning, statistics, and data analysis.
                Experience with data manipulation libraries like pandas and numpy.
                Skills in data visualization and working with big data technologies.
                Knowledge of deep learning frameworks like TensorFlow or PyTorch is a plus.
            """,
            "product_manager": """
                Product Manager with experience in product development and user research.
                Strong skills in agile methodologies, roadmap planning, and stakeholder management.
                Ability to conduct market analysis and translate business requirements into user stories.
                Experience with prioritization frameworks and product strategy.
            """
        }
        
        return descriptions.get(job_id, descriptions["software_engineer"]) 