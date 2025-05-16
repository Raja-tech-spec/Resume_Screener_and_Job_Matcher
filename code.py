import pandas as pd
import fitz  # PyMuPDF
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
import re
import time

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_CACHE_FILE = 'sentence_transformer_model.pkl'
EMBEDDINGS_CACHE_FILE = 'resume_embeddings.pkl'
DATASET_FILE = 'UpdatedResumeDataSet.csv'
JOBS_FILE = 'jobs_dataset_with_features.csv'

# Common skills and degree patterns
COMMON_SKILLS = [
    "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "PHP", "Go", "Rust", "Swift", "Kotlin",
    "HTML", "CSS", "React", "Angular", "Vue.js", "Svelte",
    "Node.js", "Django", "Flask", "Spring Boot", "Laravel", ".NET",
    "React Native", "Flutter", "Swift (iOS)", "Kotlin (Android)",
    "SQL", "PostgreSQL", "MySQL", "NoSQL", "MongoDB", "Firebase",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "GitHub Actions", "Jenkins",
    "Blockchain", "Solidity", "Web3.js", "Smart Contracts", "Ethereum",
    "AI/ML", "TensorFlow", "PyTorch", "NLP", "Computer Vision", "LLMs", "GPT", "Claude",
    "Game Dev", "Unity", "Unreal Engine", "C# for Games",
    "Cybersecurity", "Ethical Hacking", "Penetration Testing", "Cryptography",
    "Data Science", "Pandas", "NumPy", "Scikit-learn", "R",
    "Big Data", "Hadoop", "Spark", "Kafka",
    "BI Tools", "Tableau", "Power BI", "Looker",
    "Excel", "Google Sheets", "VBA", "Macros",
    "Graphic Design", "Adobe Photoshop", "Illustrator", "Canva", "Figma",
    "UI/UX Design", "Wireframing", "Prototyping", "Figma", "Adobe XD",
    "Video Editing", "Premiere Pro", "DaVinci Resolve", "After Effects",
    "3D Modeling", "Blender", "Maya", "AutoCAD",
    "Audio Production", "FL Studio", "Ableton", "Audacity",
    "Embedded Systems", "Arduino", "Raspberry Pi", "IoT",
    "Robotics", "ROS (Robot Operating System)",
    "CAD", "SolidWorks", "Fusion 360",
    "Communication", "Leadership", "Team Management",
    "Problem-Solving", "Critical Thinking",
    "Time Management", "Productivity",
    "Emotional Intelligence (EQ)", "Negotiation", "Persuasion",
    "Adaptability", "Learning Agility",
    "Public Speaking", "Presentation", "Conflict Resolution",
    "Project Management", "Agile", "Scrum", "Kanban",
    "Product Management", "Roadmapping", "MVP Development",
    "Digital Marketing", "SEO", "SEM", "Social Media Marketing", "Email Marketing",
    "Sales", "Business Development",
    "Financial Literacy", "Budgeting", "Forecasting", "Accounting Basics",
    "Entrepreneurship", "Startup Fundamentals", "Pitch Deck Creation",
    "Customer Support", "CRM", "Zendesk", "HubSpot",
    "Copywriting", "Content Writing",
    "Blogging", "SEO Writing",
    "Video Scripting", "Storytelling",
    "Photography", "Cinematography",
    "Podcasting", "Voiceovers",
    "Social Media Content Creation",
    "Academic Writing", "Research Papers",
    "Data Collection", "Surveys",
    "Statistical Analysis", "SPSS", "MATLAB",
    "Technical Documentation",
    "Language Translation",
    "Teaching", "Tutoring", "Online Courses", "Workshops",
    "Freelancing", "Remote Work Best Practices",
    "Low-Code/No-Code", "Bubble", "Webflow", "Zapier"
]

# Degree patterns and mapping for specific degrees
DEGREE_PATTERNS = [
    r'bachelor[\w\s]*\b(?:science|arts|engineering|technology)\b',
    r'b\.\w+',
    r'master[\w\s]*\b(?:science|arts|engineering|technology)\b',
    r'm\.\w+',
    r'ph\.?\s?d',
    r'doctorate',
    r'diploma',
    r'associate[\w\s]*degree',
    r'bsai',  # Bachelor of Science in Artificial Intelligence
    r'bscs',  # Bachelor of Science in Computer Science
    r'bse',   # Bachelor of Software Engineering
    r'bsit',  # Bachelor of Science in Information Technology
]

# Mapping for specific degree abbreviations to full names
DEGREE_MAPPING = {
    'bsai': 'Bachelor Student in Artificial Intelligence',
    'bscs': 'Bachelor Student in Computer Science',
    'bse': 'Bachelor Student in Software Engineering',
    'bsit': 'Bachelor Student in Information Technology',
    'bachelor of science': 'Bachelor Student in Science',
    'bachelor of arts': 'Bachelor Student in Arts',
    'bachelor of engineering': 'Bachelor Student in Engineering',
    'bachelor of technology': 'Bachelor Student in Technology',
    'master of science': 'Master Student in Science',
    'master of arts': 'Master Student in Arts',
    'master of engineering': 'Master Student in Engineering',
    'master of technology': 'Master Student in Technology',
    'ms': 'Master Student in Science',  # Default for MS if field not specified
    'ph.d': 'Doctoral Student',
    'phd': 'Doctoral Student',
    'doctorate': 'Doctoral Student',
    'diploma': 'Diploma Student',
    'associate degree': 'Associate Degree Student',
}

# Load model with caching
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_CACHE_FILE):
        with open(MODEL_CACHE_FILE, 'rb') as f:
            model = pickle.load(f)
        st.sidebar.info("Loaded model from cache")
    else:
        with st.spinner('Downloading and caching model...'):
            model = SentenceTransformer(MODEL_NAME)
            with open(MODEL_CACHE_FILE, 'wb') as f:
                pickle.dump(model, f)
        st.sidebar.info("Created and cached new model")
    return model

# Load or create embeddings
@st.cache_data
def load_or_create_embeddings(_model, resumes_df):
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
            embeddings, last_updated = pickle.load(f)
        st.sidebar.info(f"Loaded embeddings (updated {last_updated})")
    else:
        with st.spinner('Creating embeddings for all resumes...'):
            embeddings = _model.encode(resumes_df['Resume'].astype(str).tolist())
            last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
                pickle.dump((embeddings, last_updated), f)
        st.sidebar.success("Created and cached new embeddings")
    return embeddings

# Load datasets with validation
def load_datasets():
    try:
        resumes_df = pd.read_csv(DATASET_FILE)
        jobs_df = pd.read_csv(JOBS_FILE)
        if 'Category' not in resumes_df.columns or 'Resume' not in resumes_df.columns:
            st.error(f"Dataset must contain 'Category' and 'Resume' columns")
            st.stop()
        if 'Role' not in jobs_df.columns:
            st.error(f"Jobs dataset must contain 'Role' column")
            st.stop()
        return resumes_df, jobs_df
    except FileNotFoundError as e:
        st.error(f"Dataset file not found: {e.filename}")
        st.stop()

# Extract resume text from PDF
def extract_resume_text(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = " ".join(page.get_text() for page in doc)
            if not text.strip():
                st.error("PDF appears to be empty or contains no text")
                return None
            return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# Simplified skill extraction using COMMON_SKILLS
def extract_skills(resume_text):
    resume_text_lower = resume_text.lower()
    skills_found = set()
    for skill in COMMON_SKILLS:
        if re.search(rf'\b{re.escape(skill.lower())}\b', resume_text_lower):
            skills_found.add(skill.title())
    return sorted(skills_found)[:15]

# Improved education extraction (only degree name)
def extract_education(resume_text):
    resume_text_lower = resume_text.lower()
    
    # Extract degrees
    for pattern in DEGREE_PATTERNS:
        matches = re.findall(pattern, resume_text_lower, re.I)
        for match in matches:
            match_clean = match.lower().replace('.', '').replace(' ', '')
            # Look for specific degree abbreviations first
            for degree_abbr, degree_name in DEGREE_MAPPING.items():
                if degree_abbr in match_clean:
                    return degree_name
            # If no specific abbreviation matches, try to match general patterns
            for degree_key, degree_name in DEGREE_MAPPING.items():
                if degree_key in match.lower():
                    return degree_name
    
    # If no degree is found, return empty string
    return ""

# Analyze resume
def analyze_resume(uploaded_file, model, resumes_df, jobs_df, embeddings):
    progress_bar = st.progress(0)
    
    # Extract text
    progress_bar.progress(10, text="Extracting text from PDF...")
    resume_text = extract_resume_text(uploaded_file)
    if not resume_text:
        return None, None, None, None
    
    # Extract skills and education
    progress_bar.progress(30, text="Extracting skills and education...")
    skills = extract_skills(resume_text)
    education = extract_education(resume_text)
    
    # Generate embedding
    progress_bar.progress(50, text="Analyzing resume content...")
    resume_embedding = model.encode([resume_text])
    
    # Find matches
    progress_bar.progress(70, text="Finding best matches...")
    similarities = cosine_similarity(resume_embedding, embeddings)
    top_matches = np.argsort(similarities[0])[::-1][:3]
    
    # Prepare results
    progress_bar.progress(90, text="Preparing results...")
    results = []
    for idx in top_matches:
        match = {
            'category': resumes_df.iloc[idx]['Category'],
            'role': jobs_df.iloc[idx]['Role'],
            'similarity': float(similarities[0][idx]),
            'resume_sample': (resumes_df.iloc[idx]['Resume'][:200] + "...") if pd.notna(resumes_df.iloc[idx]['Resume']) else "[No content]"
        }
        results.append(match)
    
    progress_bar.progress(100, text="Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    
    return resume_text, skills, education, results

# Display analysis results
def show_analysis_results(resume_text, skills, education, results):
    st.subheader("📄 Extracted Resume Content")
    with st.expander("View resume text", expanded=False):
        st.text_area("Resume Content", resume_text, height=200, label_visibility="collapsed")
    
    # Skills and Education columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠️ Skills")
        if skills:
            container = st.container(border=True)
            container.metric(
                label="Identified Skills",
                value=f"{len(skills)} Skills Found",
                delta="Skills extracted from resume"
            )
            with container.expander("Details"):
                st.write("Here are the skills we identified in your resume:")
                for skill in skills:
                    st.markdown(f"- {skill}")
        else:
            st.warning("No skills detected in the resume")
    
    with col2:
        st.subheader("🎓 Education")
        if education:
            container = st.container(border=True)
            container.metric(
                label="Education Level",
                value=education,
                delta="Extracted from resume"
            )
            # No expander since we only want the degree name
        else:
            st.warning("No education information detected in the resume")
    
    st.markdown("---")
    st.subheader("🎯 Top Job Matches")
    
    if not results:
        st.warning("No matches found")
        return
    
    cols = st.columns(3)
    for i, match in enumerate(results[:3]):
        with cols[i]:
            container = st.container(border=True)
            container.metric(
                label=f"Match #{i+1}",
                value=match['role'],
                delta=f"{match['similarity']:.1%} match"
            )
            
            with container.expander("Details"):
                st.markdown(f"**Category:** {match['category']}")
                st.markdown("**Similar Resume Excerpt:**")
                st.info(match['resume_sample'])

# Cache management
def clear_cache():
    if st.sidebar.button("Confirm Clear Cache"):
        if os.path.exists(MODEL_CACHE_FILE):
            os.remove(MODEL_CACHE_FILE)
        if os.path.exists(EMBEDDINGS_CACHE_FILE):
            os.remove(EMBEDDINGS_CACHE_FILE)
        st.sidebar.success("Cache cleared successfully!")
        st.rerun()

def main():
    st.set_page_config(
        page_title="Resume Matcher Pro",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS to increase text size for metric value
    st.markdown("""
    <style>
    .st-emotion-cache-1v0mbdj img {
        border-radius: 8px;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }
    /* Increase the font size of the metric value to match the image */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with cache controls
    with st.sidebar:
        st.title("Settings")
        st.write(f"Model: `{MODEL_NAME}`")
        
        with st.expander("Cache Management"):
            st.warning("Clearing cache will require rebuilding models")
            clear_cache()
        
        st.markdown("---")
        st.write("Dataset Info")
        try:
            resumes_df, _ = load_datasets()
            st.write(f"Resumes loaded: {len(resumes_df)}")
            st.write(f"Categories: {len(resumes_df['Category'].unique())}")
        except:
            st.write("Dataset not loaded")
    
    # Main interface
    st.title("AI Resume-Job Matcher By Notch Developers")
    st.caption("Upload your resume to find the best matching job roles from our database")
    
    # Load data and models
    model = load_model()
    resumes_df, jobs_df = load_datasets()
    embeddings = load_or_create_embeddings(model, resumes_df)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a resume PDF",
        type="pdf",
        help="Upload your resume in PDF format"
    )
    
    if uploaded_file:
        resume_text, skills, education, results = analyze_resume(
            uploaded_file, model, resumes_df, jobs_df, embeddings
        )
        if resume_text and results:
            show_analysis_results(resume_text, skills, education, results)
    
    # Dataset info expander
    with st.expander("ℹ️ About This Tool"):
        st.write("""
        This tool uses natural language processing to:
        - Extract text from your resume
        - Identify key skills and education
        - Analyze its content using sentence embeddings
        - Match against our database of job roles
        - Show you the best matching opportunities
        """)
        st.write(f"Last model update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
