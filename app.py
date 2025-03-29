from flask import Flask, request, jsonify, render_template
from main import AutomatedInterviewSystem
import os

app = Flask(__name__)
interview_system = AutomatedInterviewSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    resume = request.files['resume']
    video = request.files['video']
    code = request.form['code']
    job_id = request.form['job_id']
    
    # Save files temporarily
    resume_path = os.path.join('uploads', resume.filename)
    video_path = os.path.join('uploads', video.filename)
    
    resume.save(resume_path)
    video.save(video_path)
    
    # Process candidate
    result = interview_system.process_candidate(
        resume_path, video_path, code, job_id
    )
    
    # Clean up
    os.remove(resume_path)
    os.remove(video_path)
    
    return jsonify(result)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True) 