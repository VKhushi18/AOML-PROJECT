import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import PyPDF2
from PIL import Image
import pytesseract
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="AI Interview Assistant", page_icon="👔", layout="wide")

# Custom CSS for better visibility
st.markdown("""
<style>
    .interview-question { background-color: #1e3a8a; color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px; }
    .option-box { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid #0068c9; }
    .feedback-box { background-color: #e6f3ff; padding: 15px; border-radius: 5px; border-left: 5px solid #0068c9; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("AI Interview Assistant")
st.sidebar.info("Upload your resume and complete the interview to get an AI-powered evaluation.")

# Helper functions for file processing
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# AI Interview System
class AIInterviewSystem:
    def __init__(self):
        # Job skills
        self.job_skills = {
            "software_engineer": ["python", "java", "javascript", "algorithms", "git", "agile"],
            "data_scientist": ["python", "r", "sql", "machine learning", "statistics", "pandas"],
            "product_manager": ["product development", "user research", "agile", "roadmap", "stakeholder"]
        }
        
        # Replace random model training with pre-defined evaluation logic
        self.model_accuracies = {
            "Random Forest": 0.85,
            "Decision Tree": 0.78,
            "SVM": 0.82,
            "KNN": 0.80,  
            "XGBoost": 0.88 if XGBOOST_AVAILABLE else 0
        }
        
        # Try to load pre-trained models
        if not self._load_models():
            # If loading fails, train new models
            self._initialize_models()
            # Save the trained models for future use
            self._save_models()
        
        # MCQ questions by role
        self.mcq_questions = {
            "software_engineer": [
                {"question": "Which data structure would be most efficient for implementing a priority queue?",
                 "options": ["Array", "Linked List", "Heap", "Hash Table"], "correct": 2},
                {"question": "What is the time complexity of quicksort in the average case?",
                 "options": ["O(n)", "O(n log n)", "O(n²)", "O(log n)"], "correct": 1},
                {"question": "Which design pattern is best suited for creating objects without specifying their concrete classes?",
                 "options": ["Singleton", "Factory", "Observer", "Decorator"], "correct": 1}
            ],
            "data_scientist": [
                {"question": "Which algorithm is NOT typically used for classification problems?",
                 "options": ["Random Forest", "Logistic Regression", "K-means", "Support Vector Machines"], "correct": 2},
                {"question": "What is the purpose of regularization in machine learning models?",
                 "options": ["Speed up training", "Prevent overfitting", "Increase model complexity", "Improve data visualization"], "correct": 1},
                {"question": "Which technique is best for handling missing values in a dataset?",
                 "options": ["Always delete rows with missing values", "Replace with mean/median", "Use predictive models to impute values", "It depends on the data and context"], "correct": 3}
            ],
            "product_manager": [
                {"question": "What is the primary purpose of a product roadmap?",
                 "options": ["Detailed technical specifications", "Timeline for feature development", "Marketing strategy", "Budget allocation"], "correct": 1},
                {"question": "Which of the following is NOT typically a product manager's responsibility?",
                 "options": ["Defining product vision", "Writing code", "Prioritizing features", "Gathering user requirements"], "correct": 1},
                {"question": "What does the acronym MVP stand for in product development?",
                 "options": ["Most Valuable Product", "Minimum Viable Product", "Maximum Value Proposition", "Multiple Version Product"], "correct": 1}
            ]
        }
        
        # Behavioral MCQs for all roles
        self.behavioral_mcqs = [
            {"question": "How do you typically handle tight deadlines?",
             "options": ["Work longer hours to complete everything myself", 
                        "Prioritize tasks and communicate clearly about what can be delivered",
                        "Cut corners to ensure everything gets done", 
                        "Delegate everything to team members"],
             "weights": [0.6, 0.9, 0.3, 0.5]},
            {"question": "When facing a conflict with a team member, what's your approach?",
             "options": ["Avoid the person to prevent further conflict", 
                        "Escalate to management immediately",
                        "Have a private conversation to understand their perspective", 
                        "Publicly address the issue to get others' input"],
             "weights": [0.2, 0.4, 0.9, 0.5]},
            {"question": "How do you handle receiving critical feedback about your work?",
             "options": ["Defend my approach and explain why I made those decisions", 
                        "Listen carefully, ask clarifying questions, and use it to improve",
                        "Accept it politely but continue with my original approach", 
                        "Compare my work with others to show it's actually good"],
             "weights": [0.5, 0.9, 0.4, 0.2]},
            {"question": "When working on a team project that's falling behind schedule, what would you do?",
             "options": ["Take on more work personally to catch up", 
                        "Analyze the bottlenecks and propose process improvements",
                        "Suggest reducing the project scope to meet the deadline", 
                        "Document the delays to protect yourself from blame"],
             "weights": [0.6, 0.9, 0.7, 0.2]}
        ]
    
    def _initialize_models(self):
        """Initialize and train models on synthetic or real data"""
        # Generate or load training data
        X_train, y_train = generate_training_data(1000)
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        
        # Initialize models
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        
        if XGBOOST_AVAILABLE:
            self.xgboost = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        
        # Train models
        self.random_forest.fit(X_train, y_train)
        self.decision_tree.fit(X_train, y_train)
        self.svm.fit(X_train, y_train)
        self.knn.fit(X_train, y_train)
        
        if XGBOOST_AVAILABLE:
            self.xgboost.fit(X_train, y_train)
        
        # Evaluate models
        y_pred = self.random_forest.predict(X_val)
        print(classification_report(y_val, y_pred))
        print(confusion_matrix(y_val, y_pred))
        
        
        param_grid = {
            'n_estimators': [50, 100],  # Reduced for faster execution
            'max_depth': [5, 10],       # Reduced for faster execution
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,  # Reduced for faster execution
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        self.random_forest = grid_search.best_estimator_
        
        # Calculate accuracies on validation set
        self.model_accuracies = {
            "Random Forest": self.random_forest.score(X_val, y_val),
            "Decision Tree": self.decision_tree.score(X_val, y_val),
            "SVM": self.svm.score(X_val, y_val),
            "KNN": self.knn.score(X_val, y_val),
            "XGBoost": self.xgboost.score(X_val, y_val) if XGBOOST_AVAILABLE else 0
        }
    
    def _predict_with_model(self, features, model_name):
        """Make predictions using trained models"""
        
        features_array = np.array(features).reshape(1, -1)
        
        # Get prediction from the appropriate model
        if model_name == "Random Forest":
            return self.random_forest.predict_proba(features_array)[0, 1]
        elif model_name == "Decision Tree":
            return self.decision_tree.predict_proba(features_array)[0, 1]
        elif model_name == "SVM":
            return self.svm.predict_proba(features_array)[0, 1]
        elif model_name == "KNN":
            return self.knn.predict_proba(features_array)[0, 1]
        elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
            return self.xgboost.predict_proba(features_array)[0, 1]
        else:
            return 0.5  # Default fallback
    
    def generate_mcq_questions(self, job_role, num_questions=6):
        """Generate a mix of technical and behavioral MCQs with behavioral questions first"""
        # Get role-specific technical questions
        tech_questions = self.mcq_questions.get(job_role, self.mcq_questions["software_engineer"])
        
        # Add category to each question
        for q in tech_questions:
            q["category"] = "Technical"
        
        # Add category to behavioral questions
        for q in self.behavioral_mcqs:
            q["category"] = "Behavioral"
        
        # Select questions from each category, ensuring we have 6 total (3 of each)
        num_tech = min(3, len(tech_questions))
        num_behavioral = min(3, len(self.behavioral_mcqs))
        
        # Select the questions
        selected_tech = random.sample(tech_questions, num_tech)
        selected_behavioral = random.sample(self.behavioral_mcqs, num_behavioral)
        
        # Combine questions with behavioral questions first, then technical
        selected_questions = selected_behavioral + selected_tech
        
        # Ensure we return exactly 6 questions or as many as possible
        return selected_questions[:min(6, len(selected_questions))]
    
    def parse_resume(self, resume_text, job_id):
        """Extract skills and experience from resume"""
        if not resume_text:
            return {"skill_match_score": 0.5, "extracted_skills": [], "confidence": 0.7, 
                    "model_predictions": {}, "model_accuracies": self.model_accuracies}
        
        # Extract skills
        skills = set()
        job_specific_skills = self.job_skills.get(job_id, [])
        for skill in job_specific_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', resume_text.lower()):
                skills.add(skill)
        
        skill_match_score = len(skills) / len(job_specific_skills) if job_specific_skills else 0.5
        
        # Get predictions using deterministic rules instead of random models
        features = [skill_match_score, 0.7, 0.7, 0.7]
        model_predictions = {
            "Random Forest": self._predict_with_model(features, "Random Forest"),
            "Decision Tree": self._predict_with_model(features, "Decision Tree"),
            "SVM": self._predict_with_model(features, "SVM"),
            "KNN": self._predict_with_model(features, "KNN")  # Add KNN prediction
        }
        
        if XGBOOST_AVAILABLE:
            model_predictions["XGBoost"] = self._predict_with_model(features, "XGBoost")
        
        return {
            "skill_match_score": skill_match_score,
            "extracted_skills": list(skills),
            "model_predictions": model_predictions,
            "model_accuracies": self.model_accuracies,
            "confidence": 0.7 + (0.3 * skill_match_score),
            "experience_years": self._extract_experience_years(resume_text),
            "education_level": self._extract_education_level(resume_text)
        }
    
    def _extract_experience_years(self, text):
        years_match = re.search(r'(\d+)\s*(year|yr)', text, re.IGNORECASE)
        return int(years_match.group(1)) if years_match else 2
    
    def _extract_education_level(self, text):
        education_level = 0.7  # Default bachelor's equivalent
        if re.search(r'\b(phd|doctor|doctorate)\b', text, re.IGNORECASE):
            education_level = 1.0
        elif re.search(r'\b(master|msc|mba)\b', text, re.IGNORECASE):
            education_level = 0.9
        return education_level
    
    def evaluate_response(self, question, response):
        """Evaluate any type of response based on question category"""
        category = question.get('category', '')
        
        if category == 'Descriptive':
            return self.evaluate_descriptive_response(question, response)
        elif 'correct' in question:  # Technical MCQ
            selected_option = int(response)
            is_correct = (selected_option == question['correct'])
            score = 1.0 if is_correct else 0.3
            
            if is_correct:
                feedback = "Correct! Great job."
            else:
                feedback = f"Not quite right. The correct answer was: {question['options'][question['correct']]}"
            
            return {
                "score": score,
                "feedback": feedback,
                "is_correct": is_correct,
                "category": category
            }
        elif 'weights' in question:  # Behavioral MCQ
            selected_option = int(response)
            score = question['weights'][selected_option]
            
            # Generate feedback based on score
            if score > 0.8:
                feedback = "Excellent response! This demonstrates strong interpersonal skills."
            elif score > 0.6:
                feedback = "Good answer. This shows solid professional judgment."
            elif score > 0.4:
                feedback = "Acceptable response, but there's room for improvement."
            else:
                feedback = "This approach might not be optimal in a professional setting."
            
            return {
                "score": score,
                "feedback": feedback,
                "category": category
            }
        
        # Default case
        return {
            "score": 0.5,
            "feedback": "Response recorded.",
            "category": category
        }
    
    def evaluate_descriptive_response(self, question, user_answer):
        """Evaluate descriptive response using NLP similarity"""
        if not user_answer or len(user_answer.strip()) < 10:
       
        
      
    
  
        
        # Get predictions from all models
        model_predictions = {
            "Random Forest": self._predict_with_model(features, "Random Forest"),
            "Decision Tree": self._predict_with_model(features, "Decision Tree"),
            "SVM": self._predict_with_model(features, "SVM"),
            "KNN": self._predict_with_model(features, "KNN")
        }
        
        if XGBOOST_AVAILABLE:
            model_predictions["XGBoost"] = self._predict_with_model(features, "XGBoost")
            ml_score = (model_predictions["Random Forest"] + 
                       model_predictions["Decision Tree"] + 
                       model_predictions["SVM"] + 
                       model_predictions["KNN"] + 
                       model_predictions["XGBoost"]) / 5
        else:
            ml_score = (model_predictions["Random Forest"] + 
                       model_predictions["Decision Tree"] + 
                       model_predictions["SVM"] + 
                       model_predictions["KNN"]) / 4
        
        # Calculate overall score
        resume_weight = 0.4
        interview_weight = 0.6
        
        overall_score = (candidate_data["resume_score"] * resume_weight) + (candidate_data["interview_score"] * interview_weight)
        
        # Combine heuristic and ML scores
        final_score = 0.7 * overall_score + 0.3 * ml_score
        
        # Generate recommendation
        if final_score > 0.8:
            recommendation = "Strong Hire"
            confidence = 0.9
        elif final_score > 0.65:
            recommendation = "Hire"
            confidence = 0.7
        elif final_score > 0.5:
            recommendation = "Consider"
            confidence = 0.6
        else:
            recommendation = "Do Not Hire"
            confidence = 0.8
        
        # Generate strengths and improvements based on actual scores
        strengths = []
        improvements = []
        
        # Resume feedback
        if candidate_data["resume_score"] > 0.7:
            strengths.append("Strong relevant skills and experience")
        elif candidate_data["resume_score"] > 0.5:
            strengths.append("Adequate skills for the position")
        else:
            improvements.append("Resume lacks sufficient relevant skills or experience")
        
        # Interview feedback - make this more accurate based on score
        if candidate_data["interview_score"] > 0.8:
            strengths.append("Excellent interview performance")
        elif candidate_data["interview_score"] > 0.65:
            strengths.append("Good interview performance")
        elif candidate_data["interview_score"] > 0.5:
            improvements.append("Interview performance was average")
        else:
            improvements.append("Interview responses need significant improvement")
        
        # ML score feedback
        if ml_score > 0.7:
            strengths.append("Strong overall profile according to ML analysis")
        elif ml_score > 0.5:
            strengths.append("Acceptable profile according to ML analysis")
        else:
            improvements.append("ML analysis suggests potential fit issues")
        
        # Add recommendation-specific feedback
        if recommendation == "Consider":
            improvements.append("Candidate shows potential but has some areas needing development")
        elif recommendation == "Do Not Hire":
            improvements.append("Overall qualifications do not align well with position requirements")
        
        return {
            "recommendation": recommendation,
            "final_score": final_score,
            "ml_score": ml_score,
            "confidence": confidence,
            "strengths": strengths,
            "improvements": improvements,
            "model_predictions": model_predictions,
            "model_accuracies": self.model_accuracies
        }

    def _save_models(self):
        """Save trained models to disk"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.random_forest, 'models/random_forest.pkl')
        joblib.dump(self.decision_tree, 'models/decision_tree.pkl')
        joblib.dump(self.svm, 'models/svm.pkl')
        joblib.dump(self.knn, 'models/knn.pkl')
        if XGBOOST_AVAILABLE:
            joblib.dump(self.xgboost, 'models/xgboost.pkl')
        
        # Save accuracies
        joblib.dump(self.model_accuracies, 'models/accuracies.pkl')

    def _load_models(self):
        """Load trained models from disk if available"""
        try:
            self.random_forest = joblib.load('models/random_forest.pkl')
            self.decision_tree = joblib.load('models/decision_tree.pkl')
            self.svm = joblib.load('models/svm.pkl')
            self.knn = joblib.load('models/knn.pkl')
            if XGBOOST_AVAILABLE:
                self.xgboost = joblib.load('models/xgboost.pkl')
            
            # Load accuracies
            self.model_accuracies = joblib.load('models/accuracies.pkl')
            return True
        except:
            return False

# Main application
def main():
    interview_system = AIInterviewSystem()
    
    # Add tab switching logic at the beginning of the app
    # Create tabs
    tab_names = ["Setup", "Interview", "Results"]
    
    # Check if we need to switch to the Results tab
    if 'active_tab' in st.session_state and st.session_state.active_tab == "Results":
        # Set the default tab index to the Results tab (index 2)
        default_tab_index = 2
        # Clear the active tab from session state to avoid getting stuck
        st.session_state.active_tab = None
    else:
        default_tab_index = 0
    
    # Create tabs with the default index
    tabs = st.tabs(tab_names)
    tab1, tab2, tab3 = tabs
    
    # Force the UI to show the Results tab if needed
    if default_tab_index == 2:
        # This is a workaround to make the Results tab active
        # We'll use JavaScript to click the Results tab
        js = f"""
        <script>
            function sleep(ms) {{
                return new Promise(resolve => setTimeout(resolve, ms));
            }}
            async function clickTab() {{
                await sleep(100);
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                if (tabs.length >= 3) {{
                    tabs[2].click();
                }}
            }}
            clickTab();
        </script>
        """
        st.components.v1.html(js, height=0)
    
    with tab1:
        st.header("Interview Setup")
        
        # Resume upload
        st.write("Upload your resume (PDF, JPG, or PNG):")
        resume_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png", "txt"])
        
        # Extract text from the uploaded file
        resume_text = ""
        if resume_file:
            with st.spinner("Processing resume..."):
                if resume_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(resume_file)
                elif resume_file.type in ["image/jpeg", "image/jpg", "image/png"]:
                    resume_text = extract_text_from_image(resume_file)
                elif resume_file.type == "text/plain":
                    resume_text = resume_file.getvalue().decode("utf-8")
                
                if resume_text:
                    st.success("Resume processed successfully!")
                else:
                    st.warning("Could not extract text from file. Please try another file.")
        
        # Job selection
        job_options = {
            "software_engineer": "Software Engineer",
            "data_scientist": "Data Scientist",
            "product_manager": "Product Manager"
        }
        
        job_id = st.selectbox(
            "Select Job Position",
            options=list(job_options.keys()),
            format_func=lambda x: job_options[x]
        )
        
        # Start interview button
        if st.button("Start Interview Process"):
            if not resume_file:
                st.error("Please upload a resume first.")
            else:
                with st.spinner("Analyzing resume and preparing interview..."):
                    # Parse resume
                    resume_analysis = interview_system.parse_resume(resume_text, job_id)
                    st.session_state.resume_analysis = resume_analysis
                    st.session_state.resume_score = resume_analysis["skill_match_score"]
                    st.session_state.experience_years = resume_analysis.get("experience_years", 2)
                    st.session_state.education_level = resume_analysis.get("education_level", 0.7)
                    
                    # Set up interview with MCQs
                    st.session_state.job_id = job_id
                    st.session_state.questions = interview_system.generate_mcq_questions(job_id)
                    st.session_state.current_question_idx = 0
                    st.session_state.responses = {}
                    st.session_state.response_analysis = []
                    st.session_state.interview_completed = False
                    
                    # Success message
                    st.success("Resume analyzed successfully!")
                    
                    # Display relevant skills instead of model accuracies
                    st.subheader("Resume Analysis")
                    
                    # Get job-specific skills
                    job_skills = interview_system.job_skills.get(job_id, [])
                    extracted_skills = resume_analysis["extracted_skills"]
                    
                    # Calculate match percentage
                    match_percentage = len(extracted_skills) / len(job_skills) * 100 if job_skills else 0
                    
                    # Display skill match percentage
                    st.metric("Skill Match", f"{match_percentage:.1f}%", f"{len(extracted_skills)} of {len(job_skills)} skills")
                    
                    # Create two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Relevant Skills Found:")
                        if extracted_skills:
                            for skill in extracted_skills:
                                st.markdown(f"✅ {skill.title()}")
                        else:
                            st.markdown("*No relevant skills found*")
                    
                    with col2:
                        st.markdown("#### Missing Skills:")
                        missing_skills = [skill for skill in job_skills if skill not in extracted_skills]
                        if missing_skills:
                            for skill in missing_skills:
                                st.markdown(f"❌ {skill.title()}")
                        else:
                            st.markdown("*All relevant skills found!*")
                    
                    st.info("Please proceed to the Interview tab")
    
    with tab2:
        st.header("AI-Powered Interview")
        
        if 'questions' not in st.session_state:
            st.warning("Please complete the setup first.")
        else:
            # Initialize question type if not already set
            if 'question_type' not in st.session_state:
                st.session_state.question_type = "Behavioral"
            
            # Add question type selector
            question_type = st.selectbox(
                "Select question type:",
                ["Behavioral", "Technical"],
                index=0 if st.session_state.question_type == "Behavioral" else 1
            )
            
            # Update session state if changed
            if question_type != st.session_state.question_type:
                st.session_state.question_type = question_type
                # Reset to first question of selected type
                all_questions = st.session_state.questions
                for i, q in enumerate(all_questions):
                    if q.get('category', '') == question_type:
                        st.session_state.current_question_idx = i
                        break
                st.rerun()
            
            # Filter questions by selected type
            all_questions = st.session_state.questions
            filtered_questions = [q for q in all_questions if q.get('category', '') == question_type]
            
            if not filtered_questions:
                st.warning(f"No {question_type} questions available.")
            else:
                # Find the current question index
                if 'current_question_idx' not in st.session_state:
                    # Initialize with the first question of the selected type
                    for i, q in enumerate(all_questions):
                        if q.get('category', '') == question_type:
                            st.session_state.current_question_idx = i
                            break
                
                current_global_idx = st.session_state.current_question_idx
                
                # Check if all questions have been answered
                all_answered = True
                for i, q in enumerate(all_questions):
                    if f"answered_{i}" not in st.session_state:
                        all_answered = False
                        break
                
                # If all questions are answered, show completion message
                if all_answered:
                    # Calculate average score
                    valid_scores = []
                    for a in st.session_state.response_analysis:
                        if a is not None and isinstance(a, dict) and "score" in a:
                            valid_scores.append(a["score"])
                    
                    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.5
                    st.session_state.interview_score = avg_score
                    st.session_state.interview_completed = True
                    
                    st.success("🎉 Interview completed successfully!")
                    st.info("Please proceed to the Results tab to see your evaluation.")
                    
                    # Add a button to go directly to results
                    if st.button("Go to Results"):
                        # Set a session state variable to indicate we should switch to the Results tab
                        st.session_state.active_tab = "Results"
                        st.rerun()
                # Make sure we have a valid question index
                elif current_global_idx >= len(all_questions):
                    st.success("All questions completed!")
                    st.info("Please proceed to the Results tab")
                else:
                    current_question = all_questions[current_global_idx]
                    
                    # If current question doesn't match selected type, find the first one that does
                    if current_question.get('category', '') != question_type:
                        for i, q in enumerate(all_questions):
                            if q.get('category', '') == question_type:
                                st.session_state.current_question_idx = i
                                current_global_idx = i
                                current_question = q
                                break
                    
                    # Find position in filtered list for progress bar
                    filtered_idx = filtered_questions.index(current_question) if current_question in filtered_questions else 0
                    
                    # Show progress
                    progress = (filtered_idx) / max(1, len(filtered_questions) - 1)
                    st.progress(progress)
                    st.write(f"{question_type} Question {filtered_idx + 1} of {len(filtered_questions)}")
                    
                    # Display current question with dark background and white text
                    st.markdown(f"""
                    <div style="background-color: #1e3a8a; color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                        <span style="background-color: {'#4CAF50' if question_type == 'Technical' else '#2196F3'}; 
                                    color: white; 
                                    padding: 3px 8px; 
                                    border-radius: 4px; 
                                    font-size: 14px;">
                            {question_type}
                        </span>
                        <h3 style="color: white;">Q: {current_question['question']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Check if this question has already been answered
                    if f"answered_{current_global_idx}" in st.session_state and st.session_state[f"answered_{current_global_idx}"]:
                        # Next question button without showing feedback
                        if st.button("Next Question", key=f"next_{current_global_idx}"):
                            # Find the next question of the same type
                            next_found = False
                            for i in range(current_global_idx + 1, len(all_questions)):
                                if all_questions[i].get('category', '') == question_type:
                                    st.session_state.current_question_idx = i
                                    next_found = True
                                    break
                            
                            # If no more questions of this type, suggest switching
                            if not next_found:
                                other_type = "Technical" if question_type == "Behavioral" else "Behavioral"
                                remaining_other_type = [q for q in all_questions if q.get('category', '') == other_type and 
                                                       f"answered_{all_questions.index(q)}" not in st.session_state]
                                
                                if remaining_other_type:
                                    st.session_state.question_type = other_type
                                    for i, q in enumerate(all_questions):
                                        if q.get('category', '') == other_type and f"answered_{i}" not in st.session_state:
                                            st.session_state.current_question_idx = i
                                            break
                                else:
                                    # All questions answered
                                    # Calculate average score
                                    valid_scores = []
                                    for a in st.session_state.response_analysis:
                                        if a is not None and isinstance(a, dict) and "score" in a:
                                            valid_scores.append(a["score"])
                                    
                                    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.5
                                    st.session_state.interview_score = avg_score
                                    st.session_state.interview_completed = True
                            
                            st.rerun()
                    else:
                        # MCQ handling for Technical and Behavioral questions
                        options = current_question['options']
                        
                        # Display options with better styling
                        st.write("### Select your answer:")
                        
                        # Use a simple radio button for options
                        selected_option = st.radio(
                            "",  # Empty label since we already have a header
                            options,
                            key=f"option_radio_{current_global_idx}"
                        )
                        
                        # Get the index of the selected option
                        option_index = options.index(selected_option)
                        
                        # Submit button for MCQ questions
                        if st.button("Submit Answer", key=f"submit_mcq_{current_global_idx}"):
                            # Evaluate response
                            with st.spinner("Analyzing your answer..."):
                                analysis = interview_system.evaluate_response(current_question, option_index)
                                
                                # Store response and analysis
                                st.session_state.responses[current_question['question']] = selected_option
                                
                                # Make sure response_analysis has enough slots
                                while len(st.session_state.response_analysis) <= current_global_idx:
                                    st.session_state.response_analysis.append(None)
                                
                                st.session_state.response_analysis[current_global_idx] = analysis
                                
                                # Mark this question as answered
                                st.session_state[f"answered_{current_global_idx}"] = True
                                
                                # Display feedback immediately
                                st.markdown(f"""
                                <div class="feedback-box" style="border: 2px solid #0068c9; border-left: 8px solid #0068c9;">
                                    <h4 style="color: #0068c9;">Feedback:</h4>
                                    <p style="color: #000000;">{analysis['feedback']}</p>
                                    <p style="color: #000000;"><strong>Score:</strong> {analysis['score']:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Add a "Next Question" button
                                if st.button("Next Question", key=f"next_after_feedback_{current_global_idx}"):
                                    st.rerun()
    
    with tab3:
        st.header("Evaluation Results")
        
        if not all(k in st.session_state for k in ['resume_score', 'interview_score', 'interview_completed']) or not st.session_state.interview_completed:
            st.warning("Please complete the interview process first.")
        else:
            # Calculate scores by category
            if st.session_state.response_analysis:
                technical_scores = [a["score"] for a in st.session_state.response_analysis 
                                   if a is not None and a.get('category', '') == 'Technical']
                behavioral_scores = [a["score"] for a in st.session_state.response_analysis 
                                    if a is not None and a.get('category', '') == 'Behavioral']
                
                # Calculate average scores for each category
                technical_score = np.mean(technical_scores) if technical_scores else 0
                behavioral_score = np.mean(behavioral_scores) if behavioral_scores else 0
                
                # Apply weights based on job role
                job_id = st.session_state.get('job_id', 'software_engineer')
                
                if job_id == 'software_engineer':
                    weights = {'Technical': 0.7, 'Behavioral': 0.3}
                elif job_id == 'data_scientist':
                    weights = {'Technical': 0.6, 'Behavioral': 0.4}
                else:  # product_manager
                    weights = {'Technical': 0.4, 'Behavioral': 0.6}
                
                # Calculate weighted interview score
                interview_score = (
                    technical_score * weights['Technical'] +
                    behavioral_score * weights['Behavioral']
                )
                
                # Update the interview score in session state
                st.session_state.interview_score = interview_score
                
                # Store category scores for display
                st.session_state.category_scores = {
                    'Technical': technical_score,
                    'Behavioral': behavioral_score
                }
            
            # Prepare candidate data
            candidate_data = {
                "resume_score": st.session_state.resume_score,
                "interview_score": st.session_state.interview_score,
                "experience_years": st.session_state.experience_years,
                "education_level": st.session_state.education_level,
                "category_scores": st.session_state.category_scores
            }
            
            # Generate recommendation
            with st.spinner("Generating final evaluation..."):
                if 'final_recommendation' not in st.session_state:
                    result = interview_system.generate_final_recommendation(candidate_data)
                    st.session_state.final_recommendation = result
                else:
                    result = st.session_state.final_recommendation
            
            # Display results
            st.subheader(f"Recommendation: {result['recommendation']}")
            st.write(f"Final Score: {result['final_score']:.2f}")
            
            # Display ML model accuracies
            st.subheader("ML Model Accuracies")
            accuracies = result["model_accuracies"]
            cols = st.columns(5)
            cols[0].metric("Random Forest", f"{accuracies['Random Forest']:.2%}")
            cols[1].metric("Decision Tree", f"{accuracies['Decision Tree']:.2%}")
            cols[2].metric("SVM", f"{accuracies['SVM']:.2%}")
            cols[3].metric("KNN", f"{accuracies['KNN']:.2%}")
            if XGBOOST_AVAILABLE:
                cols[4].metric("XGBoost", f"{accuracies['XGBoost']:.2%}")
            
            # Add ML model comparison chart
            st.subheader("ML Model Performance")

            # Create a bar chart comparing model accuracies
            accuracies = result["model_accuracies"]
            model_names = list(accuracies.keys())
            accuracy_values = [accuracies[model] for model in model_names]

            # Filter out models with zero accuracy (not available)
            filtered_models = [model for model, acc in zip(model_names, accuracy_values) if acc > 0]
            filtered_accuracies = [acc for acc in accuracy_values if acc > 0]

            # Create the comparison chart
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(filtered_models, filtered_accuracies, color=['#3366cc', '#dc3912', '#ff9900', '#109618', '#990099'])

            # Add accuracy percentage labels on top of each bar
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{filtered_accuracies[i]:.1%}',
                        ha='center', va='bottom', fontweight='bold')

            # Add labels and title
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Accuracy Score')
            ax.set_title('Machine Learning Model Accuracy Comparison')

            # Add grid lines for better readability
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)

            # Display the chart
            st.pyplot(fig)

            # Add concise explanation about the models
            st.info("""
            **ML Models Used:**
            - **Random Forest**: Evaluates overall candidate profile (Accuracy: {:.1%})
            - **Decision Tree**: Analyzes specific skills and qualifications (Accuracy: {:.1%})
            - **SVM**: Focuses on resume and interview performance (Accuracy: {:.1%})
            - **KNN**: Emphasizes experience and education factors (Accuracy: {:.1%})
            - **XGBoost**: Provides advanced pattern recognition for final scoring (Accuracy: {:.1%})
            """.format(
                accuracies['Random Forest'],
                accuracies['Decision Tree'],
                accuracies['SVM'],
                accuracies['KNN'],
                accuracies['XGBoost'] if XGBOOST_AVAILABLE else 0
            ))
            
            # Display component scores
            st.subheader("Component Scores")
            col1, col2 = st.columns(2)
            
            with col1:
                # Create a bar chart
                scores_df = pd.DataFrame({
                    'Component': ['Resume', 'Interview', 'ML Score'],
                    'Score': [
                        candidate_data["resume_score"],
                        candidate_data["interview_score"],
                        result["ml_score"]
                    ]
                })
                
                fig = plt.figure(figsize=(8, 4))
                plt.bar(scores_df['Component'], scores_df['Score'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                plt.ylim(0, 1)
                plt.ylabel('Score')
                plt.title('Component Scores')
                
                for i, v in enumerate(scores_df['Score']):
                    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
                
                st.pyplot(fig)
                
                # Add concise explanation
                st.markdown("""
                **Component Scores Explained:**
                - **Resume**: Your skill match with job requirements (higher is better)
                - **Interview**: Overall performance in technical and behavioral questions
                - **ML Score**: Prediction based on your resume, interview performance, experience, and education
                """)
            
            with col2:
                # Create a pie chart for interview performance
                if st.session_state.response_analysis:
                    # Get category scores
                    category_scores = st.session_state.get("category_scores", {})
                    technical_score = category_scores.get('Technical', 0.1)
                    behavioral_score = category_scores.get('Behavioral', 0.1)
                    
                    # Create metrics for pie chart - weighted contribution to overall score
                    performance_metrics = {
                        "Technical": max(0.1, technical_score) * weights['Technical'],
                        "Behavioral": max(0.1, behavioral_score) * weights['Behavioral']
                    }
                    
                    # Create pie chart
                    st.write("**Performance Contribution by Category:**")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    
                    # Extract labels and values for the pie chart
                    labels = list(performance_metrics.keys())
                    values = list(performance_metrics.values())
                    
                    # Normalize values to sum to 100%
                    total = sum(values)
                    values = [v/total for v in values]
                    
                    # Create a pie chart with labels outside the pie to avoid overlapping
                    wedges, texts, autotexts = ax.pie(
                        values, 
                        autopct='%1.1f%%',
                        startangle=90,
                        textprops={'fontsize': 12},
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                        colors=['#ff9999', '#66b3ff']
                    )
                    
                    # Place legend outside the pie to avoid overlapping
                    ax.legend(
                        wedges, 
                        [f"{l} (Score: {max(0.1, s):.2f})" for l, s in zip(
                            ["Technical", "Behavioral"], 
                            [technical_score, behavioral_score]
                        )],
                        title="Categories",
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1)
                    )
                    
                    # Add a title
                    ax.set_title('Weighted Performance Contribution', fontsize=14)
                    
                    # Equal aspect ratio ensures that pie is drawn as a circle
                    ax.axis('equal')
                    
                    # Display the pie chart
                    st.pyplot(fig)
                    
                    # Add a comment explaining the chart
                    st.info("""
                    **Performance Contribution:**
                    This chart shows how each category contributes to your overall evaluation based on:
                    1. Your actual score in each category
                    2. The relative importance of each category for this job role
                    
                    A larger slice indicates a stronger contribution to your final assessment.
                    """)
            
            # Display feedback
            st.subheader("Feedback Summary")
            
            # Display strengths and improvements in a single column
            if result['strengths'] or result['improvements']:
                # Strengths
                st.markdown("#### Strengths:")
                if result['strengths']:
                    for strength in result['strengths']:
                        st.markdown(f"✅ {strength}")
                else:
                    st.markdown("*No specific strengths identified*")
                
                # Areas for improvement
                st.markdown("#### Areas for Improvement:")
                if result['improvements']:
                    for improvement in result['improvements']:
                        st.markdown(f"🔍 {improvement}")
                else:
                    st.markdown("*No specific areas for improvement identified*")
            else:
                st.info("No detailed feedback available")
            
            # Export option
            if st.button("Export Report"):
                stored_result = st.session_state.final_recommendation
                report = f"""
                # Candidate Evaluation Report
                
                ## Overall Assessment
                - Final Score: {stored_result['final_score']:.2f}
                - Recommendation: {stored_result['recommendation']}
                
                ## Component Scores
                - Resume: {candidate_data['resume_score']:.2f}
                - Interview: {candidate_data['interview_score']:.2f}
                
                ## Strengths
                {chr(10).join(['- ' + s for s in stored_result['strengths']])}
                
                ## Areas for Improvement
                {chr(10).join(['- ' + i for i in stored_result['improvements']])}
                """
                
                st.download_button(
                    label="Download Report as Markdown",
                    data=report,
                    file_name="candidate_evaluation.md",
                    mime="text/markdown"
                )

            # Add a detailed score breakdown section
            st.subheader("Score Calculation Breakdown")

            # Create a DataFrame to show the calculation
            score_breakdown = pd.DataFrame({
                'Component': ['Resume Score', 'Interview Score', 'ML Model Prediction', 'Final Score'],
                'Value': [
                    f"{candidate_data['resume_score']:.2f}",
                    f"{candidate_data['interview_score']:.2f}",
                    f"{result['ml_score']:.2f}",
                    f"{result['final_score']:.2f}"
                ],
                'Weight': ['40%', '60%', '30%', '100%'],
                'Calculation': [
                    f"{candidate_data['resume_score']:.2f} × 0.4 = {candidate_data['resume_score'] * 0.4:.2f}",
                    f"{candidate_data['interview_score']:.2f} × 0.6 = {candidate_data['interview_score'] * 0.6:.2f}",
                    f"ML prediction based on all factors",
                    f"(0.7 × Overall + 0.3 × ML Score)"
                ]
            })

            # Display the breakdown table
            st.table(score_breakdown)

            # Add explanation of the calculation
            st.markdown("""
            **How Your Score Was Calculated:**

            1. **Resume Score (40%)**: Based on matching skills in your resume with job requirements
            2. **Interview Score (60%)**: Based on your performance in technical and behavioral questions
            3. **Overall Score**: Resume Score × 0.4 + Interview Score × 0.6 = {:.2f}
            4. **ML Score (30%)**: Prediction from machine learning models using all available data = {:.2f}
            5. **Final Score**: Overall Score × 0.7 + ML Score × 0.3 = {:.2f}

            The final recommendation is determined by your final score:
            - **Strong Hire**: Score > 0.8
            - **Hire**: Score > 0.65
            - **Consider**: Score > 0.5
            - **Do Not Hire**: Score ≤ 0.5
            """.format(
                candidate_data["resume_score"] * 0.4 + candidate_data["interview_score"] * 0.6,
                result["ml_score"],
                result["final_score"]
            ))

def generate_training_data(n_samples=1000):
    """Generate synthetic training data for model training"""
    np.random.seed(42)
    
    # Generate features
    resume_scores = np.random.uniform(0.3, 1.0, n_samples)
    interview_scores = np.random.uniform(0.2, 1.0, n_samples)
    experience_years = np.random.randint(0, 15, n_samples) / 10  # Normalized to 0-1
    education_levels = np.random.choice([0.6, 0.7, 0.9, 1.0], n_samples)  # Different education levels
    
    # Create feature matrix
    X = np.column_stack((resume_scores, interview_scores, experience_years, education_levels))
    
    # Generate target (hire/no hire) based on a formula
    y = (resume_scores * 0.4 + interview_scores * 0.4 + 
         experience_years * 0.1 + education_levels * 0.1)
    
    # Convert to binary outcome with some noise
    threshold = 0.65
    noise = np.random.normal(0, 0.05, n_samples)
    y = y + noise
    y_binary = (y > threshold).astype(int)
    
    return X, y_binary

if __name__ == "__main__":
    main() 