import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import asyncio
from datetime import datetime
import copy
from pdf_utils import PDFUtils  # Import the new class
from docx_utils import DocxUtils # Import the DocxUtils class
# Import your existing modules
from openai import AzureOpenAI
from llama_resume_parser import ResumeParser
from standardizer import ResumeStandardizer
from db_manager import ResumeDBManager
from OCR_resume_parser import ResumeParserwithOCR
from final_retriever import run_retriever, render_formatted_resume  # Retriever engine
from job_matcher import JobMatcher, JobDescriptionAnalyzer  # Import both classes from job_matcher
import streamlit.components.v1 as components
import uuid
import base64
# Set page configuration
st.set_page_config(
    page_title="HR Resume Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stButton button {
    width: 100%;
}
.card {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 15px;
    border-left: 5px solid #0068c9;
}
.card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.candidate-name {
    font-size: 20px;
    font-weight: bold;
    color: #0068c9;
}
.contact-info {
    color: #444;
    margin: 10px 0;
}
.score-info {
    color: #666;
    font-size: 14px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #eee;
}
.accepted {
    color: #28a745;
    font-weight: bold;
}
.rejected {
    color: #dc3545;
    font-weight: bold;
}
.result-count {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for app navigation and explanations
st.sidebar.title("HR Assistance Bot")
page = st.sidebar.selectbox("Navigate", [
    "Resume Search Engine",
    "JD-Resume Regeneration",
    "Upload & Process", 
    "Database Management", 
], index=0)  # Set index=0 to make Resume Search Engine the default

# Add explanatory text based on selected page
if page == "Resume Search Engine":
    st.sidebar.markdown("""
    """)
elif page == "JD-Resume Regeneration":
    st.sidebar.markdown("""
    ### 🎯 JD-Resume Regeneration
    This page allows you to:
    1. Input a job description
    2. Automatically extract relevant keywords
    3. Score and rank candidates based on their match
    4. View detailed matching analysis
    5. See only the most relevant candidates
    """)
elif page == "Upload & Process":
    st.sidebar.markdown("""
    ### 📤 Upload & Process
    This page allows you to:
    1. Upload multiple resume files (PDF/DOCX)
    2. Automatically process them through our pipeline:
       - Parse and extract content
       - Standardize the format
       - Store in database
    3. Preview processed resumes
    """)
elif page == "Database Management":
    st.sidebar.markdown("""
    ### 💾 Database Management
    This page enables you to:
    1. View all stored resumes
    2. Search resumes by specific fields
    3. View detailed resume information
    4. Manage the resume database
    """)

# Initialize session state for tracking job progress
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "standardizing_complete" not in st.session_state:
    st.session_state.standardizing_complete = False
if "db_upload_complete" not in st.session_state:
    st.session_state.db_upload_complete = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "standardized_files" not in st.session_state:
    st.session_state.standardized_files = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize session state for job matcher
if "matcher" not in st.session_state:
    st.session_state.matcher = None
if "extracted_keywords" not in st.session_state:
    st.session_state.extracted_keywords = set()
if "job_matcher_results" not in st.session_state:
    st.session_state.job_matcher_results = []

def process_uploaded_files(uploaded_files):
    """Process uploaded resume files through the parser"""
    st.session_state.processing_complete = False
    st.session_state.standardizing_complete = False
    st.session_state.db_upload_complete = False
    st.session_state.processed_files = []

    st.write(f"Processing {len(uploaded_files)} files...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_files = len(uploaded_files)
    processed_count = 0

    files_to_process = list(uploaded_files)

    for i, uploaded_file in enumerate(files_to_process):
        file_name = uploaded_file.name
        status_text.text(f"Processing {i+1}/{total_files}: {file_name}")

        temp_file_path = os.path.join(temp_dir, file_name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_ext = Path(file_name).suffix.lower()
        parser = ResumeParser()  # Instantiate inside loop to avoid event loop issues

        if file_ext not in parser.SUPPORTED_EXTENSIONS:
            st.warning(f"Skipping {file_name}: Unsupported file type {file_ext}")
            continue

        try:
            parsed_resume = parser.parse_resume(temp_file_path)
            if parsed_resume:
                parsed_resume["timestamp"] = datetime.now().isoformat()
                parsed_resume["original_filename"] = file_name

                output_path = parsed_dir / f"{Path(file_name).stem}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(parsed_resume, f, indent=2, ensure_ascii=False)

                st.session_state.processed_files.append(output_path)
                processed_count += 1
            else:
                st.warning(f"No content extracted from {file_name}")
        except Exception as e:
            st.error(f"Error parsing {file_name}: {str(e)}")

        progress_bar.progress((i + 1) / total_files)

    status_text.text(f"✅ Processed {processed_count}/{total_files} files")
    st.session_state.processing_complete = True
    
async def standardize_resumes():
    """Standardize the parsed resumes using ResumeStandardizer"""
    st.session_state.standardizing_complete = False
    st.session_state.standardized_files = []
    
    # Show status
    st.write(f"Standardizing {len(st.session_state.processed_files)} files...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize the standardizer
    try:
        standardizer = ResumeStandardizer()
    except ValueError as e:
        st.error(f"Error initializing standardizer: {e}")
        return
    
    total_files = len(st.session_state.processed_files)
    standardized_count = 0
    
    # Create a copy of processed files list to prevent any potential issues with iteration
    files_to_standardize = list(st.session_state.processed_files)
    
    for i, file_path in enumerate(files_to_standardize):
        status_text.text(f"Standardizing {i+1}/{total_files}: {file_path.name}")
        
        # Modify the standardizer to use our temp paths
        output_path = standardized_dir / file_path.name
        
        if output_path.exists():
            st.session_state.standardized_files.append(output_path)
            standardized_count += 1
            progress_bar.progress((i + 1) / total_files)
            continue
        
        try:
            with open(file_path, encoding="utf-8") as f:
                raw = json.load(f)
            
            content = raw.get("content", "")
            links = raw.get("links", [])
            
            if not content.strip():
                st.warning(f"Empty content in {file_path.name}, skipping.")
                continue
            
            prompt = standardizer.make_standardizer_prompt(content, links)
            raw_response = await standardizer.call_azure_llm(prompt)
            
            # Log raw response
            raw_log_path = standardized_dir / f"{file_path.stem}_raw.md"
            with open(raw_log_path, "w", encoding="utf-8") as f:
                f.write(raw_response)
            
            cleaned_json = standardizer.clean_llm_response(raw_response)
            parsed_json = json.loads(cleaned_json)
            
            # Add timestamp, file source and original filename
            parsed_json["timestamp"] = datetime.now().isoformat()
            parsed_json["source_file"] = str(file_path)
            if "original_filename" in raw:
                parsed_json["original_filename"] = raw["original_filename"]
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            
            st.session_state.standardized_files.append(output_path)
            standardized_count += 1
        except Exception as e:
            st.error(f"Error standardizing {file_path.name}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text(f"✅ Standardized {standardized_count}/{total_files} files")
    st.session_state.standardizing_complete = True

def convert_objectid_to_str(obj):
    """Recursively turn any ObjectId into its string form."""
    if isinstance(obj, dict):
        return {k: convert_objectid_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(i) for i in obj]
    # Match ObjectId by name to avoid importing bson here
    elif type(obj).__name__ == "ObjectId":
        return str(obj)
    else:
        return obj
    
def upload_to_mongodb():
    """Upload standardized resumes to MongoDB"""
    st.session_state.db_upload_complete = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize DB manager
    try:
        db_manager = ResumeDBManager()
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return
    
    total_files = len(st.session_state.standardized_files)
    uploaded_count = 0
    
    for i, file_path in enumerate(st.session_state.standardized_files):
        status_text.text(f"Uploading {i+1}/{total_files}: {file_path.name}")
        
        try:
            with open(file_path, encoding="utf-8") as f:
                resume_data = json.load(f)
            
            # Insert or update in MongoDB
            db_manager.insert_or_update_resume(resume_data)
            uploaded_count += 1
            st.session_state.uploaded_files.append(file_path.name)
        except Exception as e:
            st.error(f"Error uploading {file_path.name}: {e}")
        
        # Update progress
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text(f"✅ Uploaded {uploaded_count}/{total_files} resumes to MongoDB")
    st.session_state.db_upload_complete = True

def validate_and_reprocess_resumes(uploaded_files):
    """Validate standardized resumes and reprocess if 'name' is missing."""
    st.write("🔍 Validating standardized resumes...")
    reprocessed_count = 0

    for file_path in st.session_state.standardized_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                resume_data = json.load(f)

            # Check if 'name' is missing
            if not resume_data.get("name") or len(resume_data.get("name").split()[0]) < 2:
                st.warning(f"⚠️ Missing 'name' in {file_path.name}. Reprocessing...")
                reprocessed_count += 1

                # Find the original file in uploaded files
                original_file_name = file_path.stem  # Remove .json extension
                original_file = next(
                    (file for file in uploaded_files if Path(file.name).stem == original_file_name),
                    None
                )

                if not original_file:
                    st.error(f"❌ Original file for {file_path.name} not found in uploaded files.")
                    continue

                # Save the uploaded file temporarily
                temp_file_path = temp_dir / original_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(original_file.getbuffer())

                # Re-parse the original file
                parser = ResumeParserwithOCR()
                parsed_resume = parser.parse_resume(temp_file_path)

                if parsed_resume:
                    # Save parsed data
                    parsed_output_path = parsed_dir / f"{original_file_name}.json"
                    parser.save_to_json(parsed_resume, parsed_output_path)

                    # Re-standardize the parsed data
                    standardizer = ResumeStandardizer()
                    content = parsed_resume.get("content", "")
                    links = parsed_resume.get("links", [])
                    prompt = standardizer.make_standardizer_prompt(content, links)
                    raw_response = asyncio.run(standardizer.call_azure_llm(prompt))
                    cleaned_json = standardizer.clean_llm_response(raw_response)
                    parsed_json = json.loads(cleaned_json)

                    # Add metadata
                    parsed_json["timestamp"] = datetime.now().isoformat()
                    parsed_json["source_file"] = str(temp_file_path)
                    parsed_json["original_filename"] = original_file.name

                    # Save re-standardized data
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(parsed_json, f, indent=2, ensure_ascii=False)

                    st.success(f"✅ Reprocessed and standardized: {file_path.name}")
                else:
                    st.error(f"❌ Failed to re-parse {original_file.name}")
        except Exception as e:
            st.error(f"Error validating {file_path.name}: {e}")

    st.write(f"🔄 Reprocessed {reprocessed_count} resumes with missing 'name'.")

# Create temp directories for processing
temp_dir = Path(tempfile.gettempdir()) / "resume_processor"
parsed_dir = temp_dir / "parsed"
standardized_dir = temp_dir / "standardized"
for directory in [parsed_dir, standardized_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# -------------------
# Page: Boolean Search Engine
# -------------------
if page == "Resume Search Engine":
    # Prevent duplicate set_page_config calls in the retriever module
    original_spc = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None

    try:
        run_retriever()
    finally:
        st.set_page_config = original_spc

# -------------------
# Page: JD-Resume Regeneration
# -------------------
elif page == "JD-Resume Regeneration":
    st.title("🎯 JD-Resume Regeneration")
    st.markdown("""
    Input a job description and let our AI-powered system:
    1. Extract relevant keywords  
    2. Score candidates based on match  
    3. Show detailed analysis  
    4. Retailor their resumes on the fly  
    """)

    # Create two tabs for different search types
    tab1, tab2 = st.tabs(["🔍 Company Database Search", "👤 Individual Candidate Search"])

    with tab1:
        st.markdown("### 🔍 Search Multiple Candidates")
        st.markdown("Enter a job description to find and evaluate multiple matching candidates.")

        job_description = st.text_area(
            "📝 Enter Job Description",
            height=200,
            placeholder="Paste the job description here.",
            key="bulk_jd"
        )

        if job_description and st.button("🔍 Find Matching Candidates", type="primary", key="bulk_search"):
            progress = st.progress(0)
            status = st.empty()
            with st.spinner("Analyzing and matching candidates…"):
                matcher = JobMatcher()
                st.session_state.matcher = matcher
                analyzer = JobDescriptionAnalyzer()
                kw = analyzer.extract_keywords(job_description)
                st.session_state.extracted_keywords = kw["keywords"]
                results = matcher.find_matching_candidates(
                    job_description,
                    progress_bar=progress,
                    status_text=status
                )
                st.session_state.job_matcher_results = results
            progress.empty()
            status.empty()

        if st.session_state.extracted_keywords:
            st.subheader("🔑 Extracted Keywords")
            st.write(", ".join(sorted(st.session_state.extracted_keywords)))

        if st.session_state.job_matcher_results:
            accepted = [c for c in st.session_state.job_matcher_results if c["status"] == "Accepted"]
            if accepted:
                st.success(f"✅ Found {len(accepted)} matching candidates")
                st.subheader("📋 Detailed Candidate Profiles")
                for cand in accepted:
                    with st.expander(f"👤 {cand['name']} - Score: {cand['score']}/100", expanded=False):
                        st.markdown(f"""
                        <div class="card">
                        <div class="candidate-name">{cand['name']}</div>
                        <div class="contact-info">
                            📧 {cand['email']} | 📱 {cand['phone']}
                        </div>
                        <div class="score-info">
                            Score: {cand['score']}/100 | 
                            Status: <span class="{cand['status'].lower()}">{cand['status']}</span>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("#### 📊 Matching Analysis")
                        st.markdown(cand["reason"])

                        # Add retailor resume button
                        if st.button("🔄 Retailor Resume", key=f"retailor_bulk_{cand['mongo_id']}"):
                            with st.spinner("Retailoring resume..."):
                                safe_resume = convert_objectid_to_str(cand["resume"])
                                # Get matcher from session state
                                matcher = st.session_state.get('matcher')
                                if matcher:
                                    new_res = matcher.resume_retailor.retailor_resume(
                                        safe_resume,
                                        st.session_state.extracted_keywords,
                                        job_description
                                    )
                                if new_res:
                                        st.success("Resume retailored successfully!")
                                        # Store the retailored resume in session state
                                        st.session_state[f'resume_data_{cand["mongo_id"]}'] = new_res
                                        st.session_state[f'pdf_ready_{cand["mongo_id"]}'] = False
                                else:
                                    st.error("Error: Job matcher not initialized. Please try searching again.")

                        # Show editable form and PDF generation if resume has been retailored
                        if f'resume_data_{cand["mongo_id"]}' in st.session_state:
                            with st.form(key=f"resume_edit_form_{cand['mongo_id']}"):
                                resume_data = st.session_state[f'resume_data_{cand["mongo_id"]}']
                                resume_data["name"] = st.text_input("Name", value=resume_data.get("name", ""))
                                resume_data["title"] = st.text_input("Title", value=resume_data.get("title", ""))
                                resume_data["summary"] = st.text_area("Summary", value=resume_data.get("summary", ""), height=100)
                                
                                st.subheader("Education")
                                for i, edu in enumerate(resume_data["education"]):
                                    st.markdown(f"**Education {i+1}**")
                                    edu_col1, edu_col2, edu_col3 = st.columns(3)
                                    with edu_col1:
                                        edu["institution"] = st.text_input(f"Institution {i+1}", value=edu.get("institution", ""), key=f"institution_{cand['mongo_id']}_{i}")
                                    with edu_col2:
                                        edu["degree"] = st.text_input(f"Degree {i+1}", value=edu.get("degree", ""), key=f"degree_{cand['mongo_id']}_{i}")
                                    with edu_col3:
                                        edu["year"] = st.text_input(f"Year {i+1}", value=edu.get("year", ""), key=f"year_{cand['mongo_id']}_{i}")
                                    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
                                    with btn_col1:
                                        if st.form_submit_button(f"⬆️ Edu {i+1}"):
                                            if i > 0:
                                                resume_data["education"][i - 1], resume_data["education"][i] = resume_data["education"][i], resume_data["education"][i - 1]
                                    with btn_col2:
                                        if st.form_submit_button(f"⬇️ Edu {i+1}"):
                                            if i < len(resume_data["education"]) - 1:
                                                resume_data["education"][i + 1], resume_data["education"][i] = resume_data["education"][i], resume_data["education"][i + 1]
                                    with btn_col3:
                                        if st.form_submit_button(f"🗑️ Delete Edu {i+1}"):
                                            resume_data["education"].pop(i)
                                if st.form_submit_button("➕ Add Education"):
                                    resume_data["education"].append({"institution": "", "degree": "", "year": ""})

                                st.subheader("Certifications")
                                # Ensure certifications is a list
                                if "certifications" not in resume_data:
                                    resume_data["certifications"] = []
                                for i, cert in enumerate(resume_data["certifications"]):
                                    st.markdown(f"**Certification {i+1}**")
                                    # Handle both string and dict formats for certifications
                                    if isinstance(cert, dict):
                                        cert["title"] = st.text_input(f"Certification Title {i+1}", value=cert.get("title", ""), key=f"cert_title_{cand['mongo_id']}_{i}")
                                        cert["issuer"] = st.text_input(f"Issuing Organization {i+1}", value=cert.get("issuer", ""), key=f"cert_issuer_{cand['mongo_id']}_{i}")
                                        cert["year"] = st.text_input(f"Year {i+1}", value=cert.get("year", ""), key=f"cert_year_{cand['mongo_id']}_{i}")
                                    else:
                                        # Convert string to dict format
                                        cert_title = st.text_input(f"Certification Title {i+1}", value=str(cert), key=f"cert_title_{cand['mongo_id']}_{i}")
                                        cert_issuer = st.text_input(f"Issuing Organization {i+1}", value="", key=f"cert_issuer_{cand['mongo_id']}_{i}")
                                        cert_year = st.text_input(f"Year {i+1}", value="", key=f"cert_year_{cand['mongo_id']}_{i}")
                                        resume_data["certifications"][i] = {"title": cert_title, "issuer": cert_issuer, "year": cert_year}
                                    
                                    cert_btn_col1, cert_btn_col2, cert_btn_col3 = st.columns([1, 1, 1])
                                    with cert_btn_col1:
                                        if st.form_submit_button(f"⬆️ Cert {i+1}"):
                                            if i > 0:
                                                resume_data["certifications"][i - 1], resume_data["certifications"][i] = resume_data["certifications"][i], resume_data["certifications"][i - 1]
                                    with cert_btn_col2:
                                        if st.form_submit_button(f"⬇️ Cert {i+1}"):
                                            if i < len(resume_data["certifications"]) - 1:
                                                resume_data["certifications"][i + 1], resume_data["certifications"][i] = resume_data["certifications"][i], resume_data["certifications"][i + 1]
                                    with cert_btn_col3:
                                        if st.form_submit_button(f"🗑️ Delete Cert {i+1}"):
                                            resume_data["certifications"].pop(i)
                                if st.form_submit_button("➕ Add Certification"):
                                    resume_data["certifications"].append({"title": "", "issuer": "", "year": ""})

                                st.subheader("Projects")
                                for i, proj in enumerate(resume_data["projects"]):
                                    st.markdown(f"**Project {i+1}**")
                                    proj["title"] = st.text_input(f"Title {i+1}", value=proj.get("title", ""), key=f"title_{cand['mongo_id']}_{i}")
                                    proj["description"] = st.text_area(f"Description {i+1}", value=proj.get("description", ""), key=f"desc_{cand['mongo_id']}_{i}", height=300)
                                    proj_btn_col1, proj_btn_col2, proj_btn_col3 = st.columns([1, 1, 1])
                                    with proj_btn_col1:
                                        if st.form_submit_button(f"⬆️ Move Up {i+1}"):
                                            if i > 0:
                                                resume_data["projects"][i - 1], resume_data["projects"][i] = resume_data["projects"][i], resume_data["projects"][i - 1]
                                    with proj_btn_col2:
                                        if st.form_submit_button(f"⬇️ Move Down {i+1}"):
                                            if i < len(resume_data["projects"]) - 1:
                                                resume_data["projects"][i + 1], resume_data["projects"][i] = resume_data["projects"][i], resume_data["projects"][i + 1]
                                    with proj_btn_col3:
                                        if st.form_submit_button(f"🗑️ Delete {i+1}"):
                                            resume_data["projects"].pop(i)
                                if st.form_submit_button("➕ Add Project"):
                                    resume_data["projects"].append({"title": "", "description": ""})

                                st.subheader("Skills")
                                updated_skills = []
                                for i, skill in enumerate(resume_data["skills"]):
                                    skill_col1, skill_col2, skill_col3, skill_col4 = st.columns([4, 1, 1, 1])
                                    with skill_col1:
                                        skill_input = st.text_input(f"Skill {i+1}", value=skill, key=f"skill_input_{cand['mongo_id']}_{i}")
                                        updated_skills.append(skill_input)
                                    with skill_col2:
                                        if st.form_submit_button(f"⬆️ Skill {i+1}"):
                                            if i > 0:
                                                resume_data["skills"][i - 1], resume_data["skills"][i] = resume_data["skills"][i], resume_data["skills"][i - 1]
                                    with skill_col3:
                                        if st.form_submit_button(f"⬇️ Skill {i+1}"):
                                            if i < len(resume_data["skills"]) - 1:
                                                resume_data["skills"][i + 1], resume_data["skills"][i] = resume_data["skills"][i], resume_data["skills"][i + 1]
                                    with skill_col4:
                                        if st.form_submit_button(f" Delete Skill {i+1}"):
                                            resume_data["skills"].pop(i)
                                if st.form_submit_button("➕ Add Skill"):
                                    resume_data["skills"].append("")
                                submit_button = st.form_submit_button("Update and Generate New PDF")
                            if submit_button:
                                resume_data["skills"] = [s.strip() for s in updated_skills if s.strip()]
                                st.session_state[f'resume_data_{cand["mongo_id"]}'] = copy.deepcopy(resume_data)
                                with st.spinner("Generating PDF..."):
                                    # Always use extracted keywords from session state
                                    keywords = st.session_state.get('extracted_keywords', None)
                                    pdf_file, html_out = PDFUtils.generate_pdf(resume_data, keywords=keywords)
                                    pdf_b64 = PDFUtils.get_base64_pdf(pdf_file)
                                    st.session_state[f'generated_pdf_{cand["mongo_id"]}'] = pdf_file
                                    st.session_state[f'generated_pdf_b64_{cand["mongo_id"]}'] = pdf_b64
                                    st.session_state[f'pdf_ready_{cand["mongo_id"]}'] = True
                                    st.success("PDF generated successfully!")
# ...existing 
                        if st.session_state.get(f'pdf_ready_{cand["mongo_id"]}', False):
                            st.markdown("### 📄 Generated PDF Preview")
                            pdf_b64 = st.session_state[f'generated_pdf_b64_{cand["mongo_id"]}']
                            # pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="700" height="900" type="application/pdf"></iframe>'
                            # st.markdown(pdf_display, unsafe_allow_html=True)

                            # Info message and link to open in new tab
                            st.info("If the PDF is not viewable above, your browser may not support embedded PDF viewing.")
                            pdf_filename = f"{st.session_state.resume_data.get('name', 'resume').replace(' ', '_')}.pdf"
                            link_id = f"open_pdf_link_{uuid.uuid4().hex}"

                            components.html(f"""
                                <a id="{link_id}" style="margin:10px 0;display:inline-block;padding:8px 16px;font-size:16px;border-radius:5px;background:#0068c9;color:white;text-decoration:none;border:none;cursor:pointer;">
                                    🔗 Click here to open the PDF in a new tab
                                </a>
                                <script>
                                const b64Data = "{pdf_b64}";
                                const byteCharacters = atob(b64Data);
                                const byteNumbers = new Array(byteCharacters.length);
                                for (let i = 0; i < byteCharacters.length; i++) {{
                                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                                }}
                                const byteArray = new Uint8Array(byteNumbers);
                                const blob = new Blob([byteArray], {{type: "application/pdf"}});
                                const blobUrl = URL.createObjectURL(blob);
                                const link = document.getElementById("{link_id}");
                                link.href = blobUrl;
                                link.target = "_blank";
                                link.rel = "noopener noreferrer";
                                link.onclick = function() {{
                                    setTimeout(function(){{URL.revokeObjectURL(blobUrl)}}, 10000);
                                }};
                                </script>
                            """, height=80)
                            st.download_button(
                                "📥 Download PDF",
                                data=st.session_state[f'generated_pdf_{cand["mongo_id"]}'],
                                file_name=f"{resume_data.get('name', 'resume').replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key=f"pdf_download_{cand['mongo_id']}"
                            )

                            # --- Download Word Button ---
                            keywords = st.session_state.get('extracted_keywords', None)
                            word_file = DocxUtils.generate_docx(resume_data, keywords=keywords)
                            st.download_button(
                                "📝 Download Word",
                                data=word_file,
                                file_name=f"{resume_data.get('name', 'resume').replace(' ', '_')}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"word_download_{cand['mongo_id']}"
                            )


                            # Add generate summary, editable text area, and copy button
                            st.markdown("### 📝 Candidate Pitch Summary")
                            if st.button("✨ Generate Summary", key=f"generate_summary_{cand['mongo_id']}", use_container_width=True):
                                with st.spinner("Generating candidate summary..."):
                                    summary_prompt = (
                                        f"You are an expert HR professional. You MUST infer and assign a professional job title for the candidate based on the job description and their experience/skills. Do not leave the title blank. If unsure, use the most relevant title from the job description. Then, write a detailed, information-rich, single-paragraph professional summary (8-10 sentences) to introduce the following candidate to a client for a job opportunity. The summary should be written in third person, using formal and business-appropriate language, and should avoid any informal, overly enthusiastic, or emotional expressions. The summary must be comprehensive and cover the candidate's technical expertise, relevant experience, key achievements, major projects, technologies and frameworks used, leadership, teamwork, impact, and educational background as they pertain to the job description. Be specific about programming languages, frameworks, tools, and platforms the candidate has worked with. Mention any certifications or notable accomplishments. The summary should reflect high ethical standards and professionalism, and should not include any bullet points, excitement, or casual language. Use only facts from the provided information and do not invent or exaggerate. The summary should be suitable for inclusion in a formal client communication and should be at least 8-10 sentences long.\n\nReturn your response as a JSON object with two fields: 'title' and 'summary'.\n\nCandidate Information:\nName: {resume_data.get('name', '')}\nTitle: {resume_data.get('title', '')}\nSummary: {resume_data.get('summary', '')}\nSkills: {', '.join(resume_data.get('skills', []))}\n\nProjects:\n{json.dumps(resume_data.get('projects', []), indent=2)}\n\nEducation:\n{json.dumps(resume_data.get('education', []), indent=2)}\n\nJob Description:\n{job_description}"
                                    )
                                    try:
                                        client = AzureOpenAI(
                                            api_key=st.secrets["azure_openai"]["api_key"],
                                            api_version=st.secrets["azure_openai"]["api_version"],
                                            azure_endpoint=st.secrets["azure_openai"]["endpoint"]
                                        )
                                        response = client.chat.completions.create(
                                            model=st.secrets["azure_openai"]["deployment"],
                                            messages=[
                                                {"role": "system", "content": "You are an expert HR professional who writes compelling candidate summaries."},
                                                {"role": "user", "content": summary_prompt}
                                            ],
                                            temperature=0.7,
                                            response_format={"type": "json_object"}
                                        )
                                        result = response.choices[0].message.content.strip()
                                        try:
                                            result_json = json.loads(result)
                                            # Fallback: If title is empty, use the first line of the job description or a default
                                            title = result_json.get("title", "").strip()
                                            if not title:
                                                title = job_description.split("\n")[0].split("-")[0].split(":")[0].split(".")[0].strip()
                                                if not title:
                                                    title = "Candidate"
                                            resume_data["title"] = title
                                            st.session_state[f'candidate_summary_{cand["mongo_id"]}'] = result_json.get("summary", "")
                                        except Exception as e:
                                            st.session_state[f'candidate_summary_{cand["mongo_id"]}'] = result
                                    except Exception as e:
                                        st.error(f"Error generating summary: {str(e)}")
                            summary = st.session_state.get(f'candidate_summary_{cand["mongo_id"]}', "")
                            summary = st.text_area(
                                "Edit the summary as needed",
                                value=summary,
                                height=400,
                                key=f"summary_edit_{cand['mongo_id']}"
                            )
                            # Clean copy button using a hidden textarea inside the HTML component
                            components.html(f'''
                                <textarea id="copyText_{cand['mongo_id']}" style="position:absolute;left:-9999px;">{summary}</textarea>
                                <button style="margin-top:10px;padding:8px 16px;font-size:16px;border-radius:5px;background:#0068c9;color:white;border:none;cursor:pointer;"
                                    onclick="var copyText = document.getElementById('copyText_{cand['mongo_id']}'); copyText.style.display='block'; copyText.select(); document.execCommand('copy'); copyText.style.display='none'; alert('Copied!');">
                                    📋 Copy Summary
                                </button>
                            ''', height=60)
        elif job_description:
            st.info("👆 Click 'Find Matching Candidates' to start.")
        else:
            st.info("👆 Enter a job description to begin matching.")

    with tab2:
        st.markdown("### 👤 Search & Retailor for a Specific Candidate")
        st.markdown("Find a specific candidate and retailor their resume for a particular job.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            search_field = st.selectbox("Search by", ["Name", "Employee ID"], index=1)
        with col2:
            label = f"Enter candidate {search_field}" if search_field == "Name" else "Enter candidate Employee ID"
            search_value = st.text_input(label)

        # Always show job description input and search button
        job_description_single = st.text_area(
            "Enter Job Description for This Candidate", 
            height=150, 
            key="single_jd",
            placeholder="Paste the job description here."
        )

        if st.button("Find & Retailor for This Candidate", type="primary", key="single_search"):
            if not search_value or not job_description_single.strip():
                st.warning("Please enter both candidate info and job description.")
            else:
                from db_manager import ResumeDBManager
                db_manager = ResumeDBManager()
                # Build the query based on the selected search field
                if search_field == "Name":
                    query = {"name": {"$regex": f"^{search_value}$", "$options": "i"}}
                elif search_field == "Employee ID":
                    query = {"ID": {"$regex": f"^{search_value}$", "$options": "i"}}
                else:
                    query = {search_field.lower(): {"$regex": f"^{search_value}$", "$options": "i"}}
                results = db_manager.find(query)
                if not results:
                    st.error("No candidate found with that info.")
                else:
                    # If multiple, let user pick
                    if len(results) > 1:
                        st.warning("Multiple candidates found. Please select the correct one.")
                        options = [
                            f"{c.get('name', 'Unknown')} | {c.get('email', 'No email')} | {c.get('phone', 'No phone')}"
                            for c in results
                        ]
                        selected = st.selectbox("Select Candidate", options)
                        idx = options.index(selected)
                        candidate = results[idx]
                    else:
                        candidate = results[0]
                    
                    st.success(f"Found candidate: {candidate.get('name', 'Unknown')}")
                    analyzer = JobDescriptionAnalyzer()
                    keywords = analyzer.extract_keywords(job_description_single)
                    # Store keywords in session state for PDF generation
                    st.session_state.extracted_keywords = keywords["keywords"]
                    matcher = JobMatcher()
                    
                    with st.spinner("🔄 Retailoring resume..."):
                        retailored = matcher.resume_retailor.retailor_resume(
                            convert_objectid_to_str(candidate),
                            keywords["keywords"],
                            job_description_single
                        )
                    
                    if retailored:
                        # Initialize session state for resume data with all required fields
                        st.session_state.resume_data = retailored
                        st.session_state.resume_data.setdefault("education", [])
                        st.session_state.resume_data.setdefault("projects", [])
                        st.session_state.resume_data.setdefault("skills", [])
                        st.session_state.resume_data.setdefault("certifications", [])
                        st.session_state.resume_data.setdefault("name", "")
                        st.session_state.resume_data.setdefault("title", "")
                        st.session_state.resume_data.setdefault("summary", "")
                        st.session_state.pdf_ready_single = False

        # Editable form and PDF preview for individual candidate
        if st.session_state.get("resume_data") is not None:
            with st.form(key="resume_edit_form_single"):
                resume_data = st.session_state.resume_data
                resume_data["name"] = st.text_input("Name", value=resume_data.get("name", ""))
                resume_data["title"] = st.text_input("Title", value=resume_data.get("title", ""))
                resume_data["summary"] = st.text_area("Summary", value=resume_data.get("summary", ""), height=100)
                st.subheader("Education")
                for i, edu in enumerate(resume_data["education"]):
                    st.markdown(f"**Education {i+1}**")
                    edu_col1, edu_col2, edu_col3 = st.columns(3)
                    with edu_col1:
                        edu["institution"] = st.text_input(f"Institution {i+1}", value=edu.get("institution", ""), key=f"institution_single_{i}")
                    with edu_col2:
                        edu["degree"] = st.text_input(f"Degree {i+1}", value=edu.get("degree", ""), key=f"degree_single_{i}")
                    with edu_col3:
                        edu["year"] = st.text_input(f"Year {i+1}", value=edu.get("year", ""), key=f"year_single_{i}")
                    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
                    with btn_col1:
                        if st.form_submit_button(f"⬆️ Edu {i+1}"):
                            if i > 0:
                                resume_data["education"][i - 1], resume_data["education"][i] = resume_data["education"][i], resume_data["education"][i - 1]
                    with btn_col2:
                        if st.form_submit_button(f"⬇️ Edu {i+1}"):
                            if i < len(resume_data["education"]) - 1:
                                resume_data["education"][i + 1], resume_data["education"][i] = resume_data["education"][i], resume_data["education"][i + 1]
                    with btn_col3:
                        if st.form_submit_button(f"🗑️ Delete Edu {i+1}"):
                            resume_data["education"].pop(i)
                if st.form_submit_button("➕ Add Education"):
                    resume_data["education"].append({"institution": "", "degree": "", "year": ""})

                st.subheader("Certifications")
                # Ensure certifications is a list
                if "certifications" not in resume_data:
                    resume_data["certifications"] = []
                for i, cert in enumerate(resume_data["certifications"]):
                    st.markdown(f"**Certification {i+1}**")
                    # Handle both string and dict formats for certifications
                    if isinstance(cert, dict):
                        cert["title"] = st.text_input(f"Certification Title {i+1}", value=cert.get("title", ""), key=f"cert_title_single_{i}")
                        cert["issuer"] = st.text_input(f"Issuing Organization {i+1}", value=cert.get("issuer", ""), key=f"cert_issuer_single_{i}")
                        cert["year"] = st.text_input(f"Year {i+1}", value=cert.get("year", ""), key=f"cert_year_single_{i}")
                    else:
                        # Convert string to dict format
                        cert_title = st.text_input(f"Certification Title {i+1}", value=str(cert), key=f"cert_title_single_{i}")
                        cert_issuer = st.text_input(f"Issuing Organization {i+1}", value="", key=f"cert_issuer_single_{i}")
                        cert_year = st.text_input(f"Year {i+1}", value="", key=f"cert_year_single_{i}")
                        resume_data["certifications"][i] = {"title": cert_title, "issuer": cert_issuer, "year": cert_year}
                    
                    cert_btn_col1, cert_btn_col2, cert_btn_col3 = st.columns([1, 1, 1])
                    with cert_btn_col1:
                        if st.form_submit_button(f"⬆️ Cert {i+1}"):
                            if i > 0:
                                resume_data["certifications"][i - 1], resume_data["certifications"][i] = resume_data["certifications"][i], resume_data["certifications"][i - 1]
                    with cert_btn_col2:
                        if st.form_submit_button(f"⬇️ Cert {i+1}"):
                            if i < len(resume_data["certifications"]) - 1:
                                resume_data["certifications"][i + 1], resume_data["certifications"][i] = resume_data["certifications"][i], resume_data["certifications"][i + 1]
                    with cert_btn_col3:
                        if st.form_submit_button(f"🗑️ Delete Cert {i+1}"):
                            resume_data["certifications"].pop(i)
                if st.form_submit_button("➕ Add Certification"):
                    resume_data["certifications"].append({"title": "", "issuer": "", "year": ""})

                st.subheader("Projects")
                for i, proj in enumerate(resume_data["projects"]):
                    st.markdown(f"**Project {i+1}**")
                    proj["title"] = st.text_input(f"Title {i+1}", value=proj.get("title", ""), key=f"title_single_{i}")
                    proj["description"] = st.text_area(f"Description {i+1}", value=proj.get("description", ""), key=f"desc_single_{i}", height=300)
                    proj_btn_col1, proj_btn_col2, proj_btn_col3 = st.columns([1, 1, 1])
                    with proj_btn_col1:
                        if st.form_submit_button(f"⬆️ Move Up {i+1}"):
                            if i > 0:
                                resume_data["projects"][i - 1], resume_data["projects"][i] = resume_data["projects"][i], resume_data["projects"][i - 1]
                    with proj_btn_col2:
                        if st.form_submit_button(f"⬇️ Move Down {i+1}"):
                            if i < len(resume_data["projects"]) - 1:
                                resume_data["projects"][i + 1], resume_data["projects"][i] = resume_data["projects"][i], resume_data["projects"][i + 1]
                    with proj_btn_col3:
                        if st.form_submit_button(f"🗑️ Delete {i+1}"):
                            resume_data["projects"].pop(i)
                if st.form_submit_button("➕ Add Project"):
                    resume_data["projects"].append({"title": "", "description": ""})
                st.subheader("Skills")
                updated_skills = []
                for i, skill in enumerate(resume_data["skills"]):
                    skill_col1, skill_col2, skill_col3, skill_col4 = st.columns([4, 1, 1, 1])
                    with skill_col1:
                        skill_input = st.text_input(f"Skill {i+1}", value=skill, key=f"skill_input_single_{i}")
                        updated_skills.append(skill_input)
                    with skill_col2:
                        if st.form_submit_button(f"⬆️ Skill {i+1}"):
                            if i > 0:
                                resume_data["skills"][i - 1], resume_data["skills"][i] = resume_data["skills"][i], resume_data["skills"][i - 1]
                    with skill_col3:
                        if st.form_submit_button(f"⬇️ Skill {i+1}"):
                            if i < len(resume_data["skills"]) - 1:
                                resume_data["skills"][i + 1], resume_data["skills"][i] = resume_data["skills"][i], resume_data["skills"][i + 1]
                    with skill_col4:
                        if st.form_submit_button(f" Delete Skill {i+1}"):
                            resume_data["skills"].pop(i)
                if st.form_submit_button("➕ Add Skill"):
                    resume_data["skills"].append("")
                submit_button = st.form_submit_button("Update and Generate New PDF")
            if submit_button:
                resume_data["skills"] = [s.strip() for s in updated_skills if s.strip()]
                st.session_state.resume_data = copy.deepcopy(resume_data)
                with st.spinner("Generating PDF..."):
                    # Always use extracted keywords from session state
                    keywords = st.session_state.get('extracted_keywords', None)
                    pdf_file, html_out = PDFUtils.generate_pdf(resume_data, keywords=keywords)
                    pdf_b64 = PDFUtils.get_base64_pdf(pdf_file)
                    st.session_state.generated_pdf = pdf_file
                    st.session_state.generated_pdf_b64 = pdf_b64
                    st.session_state.pdf_ready_single = True
                    st.success("PDF generated successfully!")
        # After the form, show the PDF preview and download if available
        
        # ...existing code...
        if st.session_state.get("pdf_ready_single", False):
            st.markdown("### 📄 Generated PDF Preview")
            pdf_b64 = st.session_state.generated_pdf_b64
            # pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="700" height="900" type="application/pdf"></iframe>'
            # st.markdown(pdf_display, unsafe_allow_html=True)

            # Info message and link to open in new tab
            st.info("If the PDF is not viewable above, your browser may not support embedded PDF viewing.")
            link_id = f"open_pdf_link_{uuid.uuid4().hex}"

            components.html(f"""
                <a id="{link_id}" style="margin:10px 0;display:inline-block;padding:8px 16px;font-size:16px;border-radius:5px;background:#0068c9;color:white;text-decoration:none;border:none;cursor:pointer;">
                    🔗 Click here to open the PDF in a new tab
                </a>
                <script>
                const b64Data = "{pdf_b64}";
                const byteCharacters = atob(b64Data);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {{
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }}
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], {{type: "application/pdf"}});
                const blobUrl = URL.createObjectURL(blob);
                const link = document.getElementById("{link_id}");
                link.href = blobUrl;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.onclick = function() {{
                    setTimeout(function(){{URL.revokeObjectURL(blobUrl)}}, 10000);
                }};
                </script>
            """, height=80)
            

            st.download_button(
                "📄 Download PDF",
                data=st.session_state.generated_pdf,
                file_name=f"{st.session_state.resume_data.get('name', 'resume').replace(' ', '_')}.pdf",
                mime="application/pdf",
                key="pdf_download_single"
            )

            # --- Download Word Button ---
            keywords = st.session_state.get('extracted_keywords', None)
            word_file = DocxUtils.generate_docx(st.session_state.resume_data, keywords=keywords)
            st.download_button(
                "📝 Download Word",
                data=word_file,
                file_name=f"{st.session_state.resume_data.get('name', 'resume').replace(' ', '_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="word_download_single"
            )

            st.markdown("### 📝 Candidate Pitch Summary")
            if st.button("✨ Generate Summary", key="generate_summary_single", use_container_width=True):
                with st.spinner("Generating candidate summary..."):
                    summary_prompt = (
                        f"You are an expert HR professional. You MUST infer and assign a professional job title for the candidate based on the job description and their experience/skills. Do not leave the title blank. If unsure, use the most relevant title from the job description. Then, write a detailed, information-rich, single-paragraph professional summary (8-10 sentences) to introduce the following candidate to a client for a job opportunity. The summary should be written in third person, using formal and business-appropriate language, and should avoid any informal, overly enthusiastic, or emotional expressions. The summary must be comprehensive and cover the candidate's technical expertise, relevant experience, key achievements, major projects, technologies and frameworks used, leadership, teamwork, impact, and educational background as they pertain to the job description. Be specific about programming languages, frameworks, tools, and platforms the candidate has worked with. Mention any certifications or notable accomplishments. The summary should reflect high ethical standards and professionalism, and should not include any bullet points, excitement, or casual language. Use only facts from the provided information and do not invent or exaggerate. The summary should be suitable for inclusion in a formal client communication and should be at least 8-10 sentences long.\n\nReturn your response as a JSON object with two fields: 'title' and 'summary'.\n\nCandidate Information:\nName: {st.session_state.resume_data.get('name', '')}\nTitle: {st.session_state.resume_data.get('title', '')}\nSummary: {st.session_state.resume_data.get('summary', '')}\nSkills: {', '.join(st.session_state.resume_data.get('skills', []))}\n\nProjects:\n{json.dumps(st.session_state.resume_data.get('projects', []), indent=2)}\n\nEducation:\n{json.dumps(st.session_state.resume_data.get('education', []), indent=2)}\n\nJob Description:\n{job_description_single}"
                    )
                    try:
                        client = AzureOpenAI(
                            api_key=st.secrets["azure_openai"]["api_key"],
                            api_version=st.secrets["azure_openai"]["api_version"],
                            azure_endpoint=st.secrets["azure_openai"]["endpoint"]
                        )
                        response = client.chat.completions.create(
                            model=st.secrets["azure_openai"]["deployment"],
                            messages=[
                                {"role": "system", "content": "You are an expert HR professional who writes compelling candidate summaries."},
                                {"role": "user", "content": summary_prompt}
                            ],
                            temperature=0.7,
                            response_format={"type": "json_object"}
                        )
                        result = response.choices[0].message.content.strip()
                        try:
                            result_json = json.loads(result)
                            # Fallback: If title is empty, use the first line of the job description or a default
                            title = result_json.get("title", "").strip()
                            if not title:
                                title = job_description_single.split("\n")[0].split("-")[0].split(":")[0].split(".")[0].strip()
                                if not title:
                                    title = "Candidate"
                            st.session_state.resume_data["title"] = title
                            st.session_state.candidate_summary_single = result_json.get("summary", "")
                        except Exception as e:
                            st.session_state.candidate_summary_single = result
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
            summary = st.session_state.get("candidate_summary_single", "")
            summary = st.text_area(
                "Edit the summary as needed",
                value=summary,
                height=400,
                key="summary_edit_single"
            )
            # Clean copy button using a hidden textarea inside the HTML component
            components.html(f'''
                <textarea id="copyText_single" style="position:absolute;left:-9999px;">{summary}</textarea>
                <button style="margin-top:10px;padding:8px 16px;font-size:16px;border-radius:5px;background:#0068c9;color:white;border:none;cursor:pointer;"
                    onclick="var copyText = document.getElementById('copyText_single'); copyText.style.display='block'; copyText.select(); document.execCommand('copy'); copyText.style.display='none'; alert('Copied!');">
                    📋 Copy Summary
                </button>
            ''', height=60)

# -------------------
# Page: Upload & Process Resumes
# -------------------
# ...existing code...

elif page == "Upload & Process":
    st.title("📄 Resume Processing Pipeline")
    st.markdown("""
    ### Streamlined Resume Processing
    Upload a single PDF or DOC resume and let our AI-powered pipeline handle the rest. The system will automatically:
    1. Extract and parse content from your resume
    2. Standardize the information into a consistent format
    3. Store the processed data in our database

    **Supported formats:** PDF, DOC  
    **Note:** Employee ID is required.
    """)

    # --- Add Employee ID input box ---
    employee_id = st.text_input("Enter Employee ID (required)", key="employee_id_input")

    uploaded_file = st.file_uploader(
        "📤 Upload Resume File (PDF or DOC)", 
        type=["pdf", "doc"], 
        accept_multiple_files=False,
        key="resume_uploader",
        help="Upload a single resume file"
    )
    # Combined processing button
    if uploaded_file:
        if not employee_id.strip():
            st.warning("Please enter an Employee ID before processing.")
        else:
            if st.button("🚀 Process Resume", type="primary", use_container_width=True):
                with st.spinner("Processing resume..."):
                    # Step 1: Parse
                    process_uploaded_files([uploaded_file])
                    st.success("✅ Parsing complete!")

                    # Step 2: Standardize
                    asyncio.run(standardize_resumes())
                    st.success("✅ Standardization complete!")

                    # Step 3: Validate and reprocess if necessary
                    validate_and_reprocess_resumes([uploaded_file])

                    # Step 4: Upload to MongoDB (inject Employee ID before upload)
                    # --- Inject Employee ID into each standardized file ---
                    for file_path in st.session_state.standardized_files:
                        try:
                            with open(file_path, "r+", encoding="utf-8") as f:
                                data = json.load(f)
                                data["employee_id"] = employee_id.strip()
                                f.seek(0)
                                json.dump(data, f, indent=2, ensure_ascii=False)
                                f.truncate()
                        except Exception as e:
                            st.error(f"Error adding Employee ID to {file_path.name}: {e}")
                    upload_to_mongodb()
                    st.success("✅ Database upload complete!")
    else:
        st.info("👆 Please upload a PDF resume file to begin processing")

    # Display processing status
    st.subheader("📊 Processing Status")
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        if st.session_state.processing_complete:
            st.success(f"✅ Parsed {len(st.session_state.processed_files)} file(s)")
        else:
            st.info("⏳ Waiting for parsing...")
    with status_col2:
        if st.session_state.standardizing_complete:
            st.success(f"✅ Standardized {len(st.session_state.standardized_files)} file(s)")
        elif st.session_state.processing_complete:
            st.info("⏳ Ready to standardize")
        else:
            st.info("⏳ Waiting for parsing...")
    with status_col3:
        if st.session_state.db_upload_complete:
            st.success(f"✅ Uploaded {len(st.session_state.uploaded_files)} file(s) to MongoDB")
        elif st.session_state.standardizing_complete:
            st.info("⏳ Ready to upload to MongoDB")
        else:
            st.info("⏳ Waiting for standardization...")

    # Display file preview if processed
    # ...existing code...
    if st.session_state.standardized_files:
        st.subheader("👀 Preview Processed Resume")
        selected_file = st.selectbox(
            "Select a resume to preview", 
            options=[f.name for f in st.session_state.standardized_files]
        )
        if selected_file:
            file_path = standardized_dir / selected_file
            with open(file_path, "r", encoding="utf-8") as f:
                resume_data = json.load(f)
            st.markdown("---")
            # Show Employee ID in the preview
            st.info(f"**Employee ID:** {resume_data.get('employee_id', 'N/A')}")
            render_formatted_resume(resume_data)
# ...existing code...

# -------------------
# Page: Database Management
# -------------------
elif page == "Database Management":
    st.title("💾 Resume Database Management")
    
    st.markdown("""
    ### Database Operations
    
    #### Available Operations:
    1. - See complete list of candidates in database
       - View detailed information in table format
    
    2. **Search Candidates By:**
       - Name
       - Employee ID 
       - Location
       - College/University
    """)


    try:
        db_manager = ResumeDBManager()
        query_type = st.radio("Select Query Type", ["View All Resumes", "Search by Field"])
        
        if query_type == "View All Resumes":
            if "all_resumes_results" not in st.session_state:
                st.session_state.all_resumes_results = []
            if st.button("📥 Fetch All Resumes", use_container_width=True) or st.session_state.all_resumes_results:
                with st.spinner("Fetching resumes..."):
                    if not st.session_state.all_resumes_results:
                        st.session_state.all_resumes_results = db_manager.find({})
                    results = st.session_state.all_resumes_results
                    st.success(f"Found {len(results)} resumes")
                    if results:
                        resume_data = []
                        for res in results:
                            resume_data.append({
                                "employee_id": res.get("employee_id", "N/A"),
                                "Name": res.get("name", "N/A"),
                                "Email": res.get("email", "N/A"),
                                "Skills": ", ".join(res.get("skills", [])[:3]) + ("..." if len(res.get("skills", [])) > 3 else "")
                            })
                        st.dataframe(resume_data, use_container_width=True)
                        
                        resume_options = []
                        st.session_state.resume_display_map = {}
                        for res in results:
                            display_text = f"{res.get('name', 'Unknown')} - {res.get('email', 'No email')}"
                            resume_options.append(display_text)
                            st.session_state.resume_display_map[display_text] = res
                        
                        selected_resume_option = st.selectbox(
                            "Select resume to view details", 
                            options=resume_options if resume_options else ["No resumes found"],
                            key="resume_selector"
                        )
                        
                        if selected_resume_option and "No resumes found" not in selected_resume_option:
                            selected_resume = st.session_state.resume_display_map.get(selected_resume_option)
                            if selected_resume:
                                st.markdown("---")
                                render_formatted_resume(selected_resume)

                                # Add delete button with confirmation logic
                                if "delete_confirmation" not in st.session_state:
                                    st.session_state.delete_confirmation = False

                                if not st.session_state.delete_confirmation:
                                    if st.button("🗑️ Delete Resume", key="delete_button"):
                                        st.session_state.delete_confirmation = True
                                else:
                                    # Simulate a pop-up-like experience
                                    with st.container():
                                        st.error("⚠️ Are you sure you want to delete this resume? This action cannot be undone.")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.button("Yes, Delete", key="confirm_delete_button"):
                                                try:
                                                    db_manager.delete_resume({"_id": selected_resume["_id"]})
                                                    st.success(f" Deleted resume: {selected_resume.get('name', 'Unknown')}")
                                                    # Refresh the results after deletion
                                                    st.session_state.all_resumes_results = db_manager.find({})
                                                    st.session_state.delete_confirmation = False  # Reset confirmation state
                                                except Exception as e:
                                                    st.error(f"Error deleting resume: {e}")
                                        with col2:
                                            if st.button("Cancel", key="cancel_delete_button"):
                                                st.info("Deletion canceled.")
                                                st.session_state.delete_confirmation = False  # Reset confirmation state
                            else:
                                st.error("Could not find the selected resume. Please try again.")
        
        elif query_type == "Search by Field":
            col1, col2 = st.columns(2)
            with col1:
                search_field = st.selectbox(
                "Search Field", 
                    ["Name","Employee_ID","Location", "College"]
            )
            with col2:
                search_value = st.text_input("Search Value")
            
            if st.button("🔍 Search", use_container_width=True):
                if search_value:
                    query = {}
                    if search_field == "Name":
                        search_field= "name"
                        query = {search_field: {"$regex": search_value, "$options": "i"}}
                    if search_field == "Employee_ID":
                        search_field = "employee_id"
                        query = {search_field: {"$regex": search_value, "$options": "i"}}
                    if search_field == "Location":
                        search_field = "location"
                        query = {search_field: {"$regex": search_value, "$options": "i"}}
                    if search_field == "College":
                        search_field = "education.institution"
                        # Define all special institutes and their variants
                        special_institutes = {
                            "iit": [
                                "IIT",
                                "Indian Institute of Technology",
                                "Indian Inst of Technology",
                                "Indian Inst. of Technology",
                                "Indian Institute Technology",
                                "Indian Inst Technology"
                            ],
                            "iim": [
                                "IIM",
                                "Indian Institute of Management",
                                "Indian Inst of Management",
                                "Indian Inst. of Management",
                                "Indian Institute Management",
                                "Indian Inst Management"
                            ],
                            "iiit": [
                                "IIIT",
                                "Indian Institute of Information Technology",
                                "Indian Inst of Information Technology",
                                "Indian Inst. of Information Technology",
                                "Indian Institute Information Technology",
                                "Indian Inst Information Technology"
                            ],
                            "nit": [
                                "NIT",
                                "National Institute of Technology",
                                "National Inst of Technology",
                                "National Inst. of Technology",
                                "National Institute Technology",
                                "National Inst Technology"
                            ]
                        }
                        search_val_norm = search_value.strip().lower()
                        matched = None
                        for key, variants in special_institutes.items():
                            if any(search_val_norm == v.lower() for v in variants):
                                matched = key
                                break

                        if matched:
                            regex_parts = []
                            for variant in special_institutes[matched]:
                                if variant.upper() == matched.upper():
                                    regex_parts.append(rf"(^|\s){variant}(\s|$)")
                                else:
                                    regex_parts.append(variant)
                            regex_pattern = "(" + "|".join(regex_parts) + ")"
                            query = {search_field: {"$regex": regex_pattern, "$options": "i"}}
                        else:
                            query = {search_field: {"$regex": f"(^|\\s){search_value}(\\s|$)", "$options": "i"}}
                    elif "." in search_field:
                        query = {search_field: {"$regex": search_value, "$options": "i"}}
                    else:
                        query = {search_field: {"$regex": search_value, "$options": "i"}}
                    
                    with st.spinner("Searching..."):
                        results = db_manager.find(query)
                        if results:
                            st.success(f"Found {len(results)} matching resumes")
                            search_options = []
                            search_map = {}
                            for res in results:
                                display_text = f"{res.get('name', 'Unknown')} - {res.get('email', 'No email')}"
                                search_options.append(display_text)
                                search_map[display_text] = res
                            
                            # Store search results and map in session state
                            st.session_state.search_options = search_options
                            st.session_state.search_map = search_map
                        else:
                            st.warning("No matching resumes found")
                else:
                    st.warning("Please enter a search value")
            
            if "search_options" in st.session_state and st.session_state.search_options:
                selected_search_result = st.selectbox(
                    "Select resume to view details", 
                    options=st.session_state.search_options,
                    key="search_selector"
                )
                
                if selected_search_result:
                    selected_resume = st.session_state.search_map.get(selected_search_result)
                    if selected_resume:
                        st.markdown("---")
                        render_formatted_resume(selected_resume)

                        # Add delete button with confirmation logic
                        if "delete_confirmation" not in st.session_state:
                            st.session_state.delete_confirmation = False

                        if not st.session_state.delete_confirmation:
                            if st.button("🗑️ Delete Resume", key="delete_button"):
                                st.session_state.delete_confirmation = True
                        else:
                            # Simulate a pop-up-like experience
                            with st.container():
                                st.error("⚠️ Are you sure you want to delete this resume? This action cannot be undone.")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Yes, Delete", key="confirm_delete_button"):
                                        try:
                                            db_manager.delete_resume({"_id": selected_resume["_id"]})
                                            st.success(f"✅ Deleted resume: {selected_resume.get('name', 'Unknown')}")
                                            # Refresh the results after deletion
                                            st.session_state.search_options = []
                                            st.session_state.search_map = {}
                                            st.session_state.delete_confirmation = False  # Reset confirmation state
                                        except Exception as e:
                                            st.error(f"Error deleting resume: {e}")
                                with col2:
                                    if st.button("Cancel", key="cancel_delete_button"):
                                        st.info("Deletion canceled.")
                                        st.session_state.delete_confirmation = False  # Reset confirmation state
                    else:
                        st.error("Could not find the selected resume. Please try again.")

    except Exception as e:
        st.error(f"Error connecting to database: {e}")

def job_matcher_page():
    st.title("JD-Resume Regeneration")
    
    # Initialize session state variables
    if 'job_matcher_results' not in st.session_state:
        st.session_state.job_matcher_results = []
    if 'extracted_keywords' not in st.session_state:
        st.session_state.extracted_keywords = set()
    
    # Job description input
    job_description = st.text_area("Enter Job Description", height=200)
    
    if st.button("Find Matching Candidates"):
        if not job_description.strip():
            st.error("Please enter a job description")
            return
            
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Find matching candidates
        matcher = JobMatcher()
        st.session_state.job_matcher_results = matcher.find_matching_candidates(
            job_description, 
            progress_bar=progress_bar,
            status_text=status_text
        )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
    # Display results if available
    if st.session_state.job_matcher_results:
        st.write("### Matching Candidates")
        
        # Create a table for all candidates
        candidates_data = []
        for candidate in st.session_state.job_matcher_results:
            candidates_data.append({
                "Name": candidate["name"],
                "Score": f"{candidate['score']}%",
                "Status": candidate["status"],
                "Phone": candidate["phone"],
                "Email": candidate["email"]
            })
        
        # Display the table
        st.dataframe(candidates_data)
        
        # Display detailed candidate cards
        st.write("### Candidate Details")
        for candidate in st.session_state.job_matcher_results:
            with st.expander(f"{candidate['name']} - Score: {candidate['score']}%"):
                # Contact information
                st.write(f"**Phone:** {candidate['phone']}")
                st.write(f"**Email:** {candidate['email']}")
                
                # Create two columns for buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 View Matching Analysis", key=f"analysis_{candidate['mongo_id']}"):
                        st.write("**Matching Analysis:**")
                        st.write(candidate['reason'])
                
                with col2:
                    if st.button("🔄 Retailor Resume", key=f"retailor_{candidate['mongo_id']}"):
                        with st.spinner("Retailoring resume..."):
                            safe_resume = convert_objectid_to_str(candidate["resume"])
                            # Get matcher from session state
                            matcher = st.session_state.get('matcher')
                            if not matcher:
                                st.error("Error: Job matcher not initialized. Please try searching again.")
                                return
                            new_res = matcher.resume_retailor.retailor_resume(
                            safe_resume,
                            st.session_state.extracted_keywords
                        )
                            if new_res:
                                st.success("Resume retailored successfully!")
                                # Create a downloadable JSON file
                                json_str = json.dumps(new_res, indent=2)
                                st.download_button(
                                    label="📥 Download Retailored Resume",
                                    data=json_str,
                                    file_name=f"retailored_resume_{candidate['name'].replace(' ', '_')}.json",
                                    mime="application/json",
                                    key=f"download_{candidate['mongo_id']}"
                                )
                                # Show the retailored resume
                                st.write("### Retailored Resume")
                                st.json(new_res)
                
                # Show full resume button
                if st.button("📄 View Full Resume", key=f"view_{candidate['mongo_id']}"):
                    st.json(candidate['resume'])