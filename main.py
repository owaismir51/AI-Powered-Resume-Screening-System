import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import google.generativeai as genai
import os
import json
import numpy as np
from tempfile import NamedTemporaryFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


GEMINI_API_KEY = "AIzaSyAc7AMvRsyIoKWoYDWpDcvNDe4EGCYn4RI"


genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")

# Directory to store uploaded resumes
RESUME_FOLDER = "uploaded_resumes"
os.makedirs(RESUME_FOLDER, exist_ok=True)

# Dictionary to store extracted resume texts
resume_texts = {}
if 'selected_top_resumes' not in st.session_state:
    st.session_state.selected_top_resumes = []
if 'selected_indices' not in st.session_state:
    st.session_state.selected_indices = []

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def analyze_resume_with_tfidf(resume_texts, job_description):
    """Matches resumes with job description using TF-IDF and assigns a score."""
    documents = [job_description] + list(resume_texts.values())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    scores = similarity_scores.flatten() * 100  # Convert to percentage
    return scores

def evaluate_resume_with_gemini(resume_text, job_description):
    """Uses Gemini AI to evaluate and score resumes based on job description."""
    prompt = (
        "You are an expert HR interviewer. Evaluate the following resume based on the given job description. "
        "Consider the required skills and responsibilities mentioned in the job description. "
        "Provide a score from 0 to 100, where 100 means a perfect match.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Resume Content:\n{resume_text}\n\n"
        "Give only a numerical score in response."
    )
    response = model.generate_content(prompt)
    try:
        score = float(response.text.strip())
    except ValueError:
        score = 0.0  # Default score if parsing fails
    return score

def reevaluate_selected_resumes(selected_resumes, job_description):
    """Re-evaluates selected resumes considering extra beneficial skills without reducing the score."""
    updated_resumes = []
    for resume in selected_resumes:
        prompt = (
            "You are an expert HR and hiring manager specializing in resume evaluation. Your task is to "
            "**re-evaluate** the following resume in relation to the given job description.\n\n"
            "**Objective:**\n"
            "- Identify **any additional beneficial skills** beyond the basic job requirements that might be valuable for the role.\n"
            "- If such extra skills are found, **increase** the original score accordingly.\n"
            "- If no significant additional skills are found, **keep the score unchanged** (do not decrease it).\n"
            "- Ensure the scoring scale remains between 0-100.\n\n"
            "**Guidelines:**\n"
            "1. Consider technical skills, certifications, project experience, leadership qualities, domain expertise, "
            "or unique qualifications that align with or enhance the job role.\n"
            "2. If no relevant extra skills are found, return the same score as before.\n"
            "3. Provide **only the final numerical score** as output (no explanations).\n\n"
            "---\n"
            f"**Job Description:**\n{job_description}\n\n"
            f"**Original Resume Score:** {resume['original_score']}\n\n"
            f"**Resume Content:**\n{resume['text']}\n\n"
            "**Final Score (Only return a number, do not provide explanations):**"
        )
        
        response = model.generate_content(prompt)
        try:
            new_score = float(response.text.strip())
            if new_score < resume['original_score']:  # Ensure score never decreases
                new_score = resume['original_score']
        except ValueError:
            new_score = resume['original_score']  # Fallback to original score
        
        updated_resumes.append({**resume, "score": round(new_score, 2)})
    return updated_resumes

# Streamlit UI
st.title("AI-Powered Resume Screening System")

uploaded_files = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type=["pdf"])
job_description = st.text_area("Enter Job Description")

if st.button("Process Resumes") and uploaded_files and job_description:
    resume_texts.clear()
    resume_scores = []
    
    # Process uploaded files
    for uploaded_file in uploaded_files:
        # Save file and extract text
        file_path = os.path.join(RESUME_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        resume_text = extract_text_from_pdf(file_path)
        
        # Get AI evaluation score
        gemini_score = evaluate_resume_with_gemini(resume_text, job_description)
        resume_scores.append({
            "filename": uploaded_file.name,
            "text": resume_text,
            "score": round(gemini_score, 2),
            "original_score": round(gemini_score, 2)
        })

    # Sort and display initial results
    resume_scores.sort(key=lambda x: x["score"], reverse=True)
    st.session_state.resume_scores = resume_scores

# Display results and selection interface
if 'resume_scores' in st.session_state:
    st.subheader("Initial Ranking Results")
    
    # Create checkboxes for selection
    selected_indices = []
    cols = st.columns(4)
    for idx, resume in enumerate(st.session_state.resume_scores):
        with cols[idx % 4]:
            if st.checkbox(
                f"{resume['filename']} ({resume['score']})",
                key=f"select_{idx}",
                help="Select for re-evaluation"
            ):
                selected_indices.append(idx)

    # Re-evaluation section
    if selected_indices:
        if st.button("Re-evaluate Selected Resumes"):
            selected_resumes = [st.session_state.resume_scores[i] for i in selected_indices]
            updated_resumes = reevaluate_selected_resumes(selected_resumes, job_description)
            
            # Update main list with new scores
            for updated in updated_resumes:
                for idx, original in enumerate(st.session_state.resume_scores):
                    if original["filename"] == updated["filename"]:
                        st.session_state.resume_scores[idx] = updated
            
            st.success("Resumes re-evaluated successfully!")

    # Display final ranking with color coding
    st.subheader("Final Ranking")
    for idx, resume in enumerate(sorted(st.session_state.resume_scores, 
                                      key=lambda x: x["score"], reverse=True), 1):
        score_diff = resume["score"] - resume["original_score"]
        color = "green" if score_diff > 0 else "gray"
        
        st.markdown(
            f"**{idx}. {resume['filename']}** "
            f"<span style='color:{color}'>({resume['score']} | Δ{score_diff:+.1f})</span>",
            unsafe_allow_html=True
        )
