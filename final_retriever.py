import json
import re
import streamlit as st
from pymongo import MongoClient
from boolean.boolean import BooleanAlgebra, Symbol, AND, OR
import config


class BooleanSearchParser:
    def __init__(self):
        self.algebra = BooleanAlgebra()
        self.quoted_phrases = {}
        self.placeholder_counter = 0

    def normalize_operator(self, token: str) -> str:
        """Normalize boolean operators to uppercase regardless of input case."""
        token_upper = token.upper()
        if token_upper in {"AND", "OR", "NOT"}:
            return token_upper
        return token

    def preprocess_query(self, query: str) -> str:
        """Extract quoted phrases, normalize operators, insert implicit ANDs."""
        # Step 1: extract and replace quoted phrases (handles single or double quotes)
        def replace_quoted(match):
            phrase = match.group(1) or match.group(2)
            placeholder = f"QUOTED_PHRASE_{self.placeholder_counter}"
            self.quoted_phrases[placeholder] = phrase
            self.placeholder_counter += 1
            return placeholder

        text = re.sub(
                r"\"([^\\\"]+)\"|'([^']+)'",
                replace_quoted,
                query
            )

        # Step 2: split into tokens (operators, parentheses, placeholders, words)
        tokens = re.findall(r"\(|\)|QUOTED_PHRASE_\d+|\w+", text)
        result = []
        ops = {"and", "or", "not"}  # Keep operators lowercase for now

        # Step 3: insert implicit AND between adjacent bare terms
        for i, tok in enumerate(tokens):
            result.append(tok)
            if i < len(tokens) - 1:
                nxt = tokens[i+1]
                if tok.lower() not in ops and tok not in ['(', ')'] and \
                   nxt.lower() not in ops and nxt not in ['(', ')']:
                    result.append('and')  # Use lowercase 'and' for implicit AND

        return ' '.join(result)

    def parse_query(self, query: str):
        # Reset parser state
        self.quoted_phrases = {}
        self.placeholder_counter = 0

        # Preprocess (handles quoted phrases and implicit ANDs)
        processed = self.preprocess_query(query)
        
        # Convert boolean operators to uppercase just before parsing
        processed = re.sub(r'\b(and|or|not)\b', lambda m: m.group(1).upper(), processed, flags=re.IGNORECASE)
        
        # Finally parse into Boolean expression tree
        return self.algebra.parse(processed)





def normalize(text: str) -> str:
    """Lowercase, split CamelCase, remove noise, then inject merged bigrams & halves."""
    # 1) Split CamelCase: "HuggingFace" → "Hugging Face"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # 2) Lowercase & basic cleanup
    text = text.lower()
    text = re.sub(r'(?<![\w@])\.net(?![\w.])', ' dotnet ', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3) Capture quoted phrases in the _source text_ and append the no-space form
    quoted_phrases = re.findall(r'"([^"]+)"', text)
    for phrase in quoted_phrases:
        text += " " + phrase.replace(" ", "")

    # 4) Strip quotation marks, remove symbols
    text = text.replace('"', '')
    text = re.sub(r'[^\w\s]', ' ', text)

    # 5) Tokenize & inject bigrams + halves
    words = text.split()
    # 5a) adjacent-word bigrams
    for i in range(len(words) - 1):
        text += " " + words[i] + words[i+1]
    # 5b) for long merged tokens, also inject a halved split
    for tok in words:
        if len(tok) > 8:
            mid = len(tok) // 2
            text += f" {tok[:mid]} {tok[mid:]}"

    # 6) Normalize whitespace and ensure single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7) Remove duplicate words
    words = text.split()
    unique_words = []
    seen_words = set()
    for word in words:
        if word not in seen_words:
            seen_words.add(word)
            unique_words.append(word)
    
    return ' '.join(unique_words)

# Flattener
def flatten_json(obj) -> str:
    parts = []
    def recurse(x):
        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                recurse(v)
        elif isinstance(x, list):
            for i in x:
                recurse(i)
    recurse(obj)
    return " ".join(parts)

# Evaluator

def evaluate_expression(expr, text, quoted_phrases=None):
    """Recursively evaluate Boolean expression against text, with substring fallback."""
    quoted_phrases = quoted_phrases or {}
    
    if isinstance(expr, Symbol):
        term = str(expr.obj).lower()
        
        # 1) exact‐phrase placeholders
        if term.startswith("QUOTED_PHRASE_") and term in quoted_phrases:
            phrase = quoted_phrases[term].lower()
            return phrase in text
        
        # 2) Special handling for short terms (4 or fewer characters)
        if len(term) <= 4:
            # For short terms, require exact word boundary match
            pattern = r'\b' + re.escape(term) + r'\b'
            return bool(re.search(pattern, text))
        
        # 3) For longer terms, use word boundary match
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text):
            return True
        
        # 4) fallback: substring (only for terms longer than 4 characters)
        return term in text

    elif isinstance(expr, AND):
        # For AND operations, ensure all terms are found in different positions
        found_positions = set()
        for arg in expr.args:
            if isinstance(arg, Symbol):
                term = str(arg.obj).lower()
                if term.startswith("QUOTED_PHRASE_") and term in quoted_phrases:
                    phrase = quoted_phrases[term].lower()
                    pos = text.find(phrase)
                    if pos == -1:
                        return False
                    found_positions.add(pos)
                else:
                    pattern = r'\b' + re.escape(term) + r'\b'
                    match = re.search(pattern, text)
                    if not match:
                        return False
                    found_positions.add(match.start())
            else:
                if not evaluate_expression(arg, text, quoted_phrases):
                    return False
        return True

    elif isinstance(expr, OR):
        # For OR operations, check if any term matches
        for arg in expr.args:
            if evaluate_expression(arg, text, quoted_phrases):
                return True
        return False
    
    return False

def extract_search_terms(expr, quoted_phrases=None):
    """Extract all search terms from the boolean expression for highlighting."""
    quoted_phrases = quoted_phrases or {}
    terms = set()
    
    if isinstance(expr, Symbol):
        term = str(expr.obj).lower()
        if term.startswith("QUOTED_PHRASE_") and term in quoted_phrases:
            terms.add(quoted_phrases[term].lower())
        else:
            terms.add(term)
    elif isinstance(expr, (AND, OR)):
        for arg in expr.args:
            terms.update(extract_search_terms(arg, quoted_phrases))
    
    return terms

def highlight_text(text: str, matched_terms: set) -> str:
    """Highlight matched terms in text using HTML while preserving original case."""
    if not matched_terms or not text:
        return text
    
    # Sort terms by length (longest first) to avoid partial matches
    sorted_terms = sorted(matched_terms, key=len, reverse=True)
    
    # Create a copy of the text to modify
    highlighted_text = text
    
    # Replace each term with its highlighted version
    for term in sorted_terms:
        # Escape special characters in the term for regex
        escaped_term = re.escape(term)
        # Use word boundaries to match whole words only, case-insensitive
        pattern = r'\b' + escaped_term + r'\b'
        
        # Find all matches with their original case
        matches = list(re.finditer(pattern, highlighted_text, flags=re.IGNORECASE))
        
        # Process matches in reverse order to maintain correct positions
        for match in reversed(matches):
            original_text = match.group(0)  # Get the matched text with original case
            highlighted_text = highlighted_text[:match.start()] + \
                             f'<span style="background-color: #ffeb3b; font-weight: bold;">{original_text}</span>' + \
                             highlighted_text[match.end():]
    
    return highlighted_text

def highlight_dict_values(d: dict, matched_terms: set) -> dict:
    """Recursively highlight text in dictionary values."""
    result = {}
    for key, value in d.items():
        if isinstance(value, str):
            result[key] = highlight_text(value, matched_terms)
        elif isinstance(value, dict):
            result[key] = highlight_dict_values(value, matched_terms)
        elif isinstance(value, list):
            result[key] = [highlight_text(str(item), matched_terms) if isinstance(item, str) 
                          else highlight_dict_values(item, matched_terms) if isinstance(item, dict)
                          else item for item in value]
        else:
            result[key] = value
    return result

def display_json(data):
    if "_id" in data:
        del data["_id"]
    
    st.json(data)

def render_formatted_resume(resume: dict):
    st.subheader(f"{resume.get('name', 'Candidate')} - Profile")
    # Basic information
    st.markdown("### 👤 Basic Information")
    col1, col2 = st.columns(2)
    col1.markdown(f"**Name:** {resume.get('name', 'N/A')}", unsafe_allow_html=True)
    col1.markdown(f"**Email:** {resume.get('email', 'N/A')}", unsafe_allow_html=True)
    col2.markdown(f"**Phone:** {resume.get('phone', 'N/A')}", unsafe_allow_html=True)
    col2.markdown(f"**Location:** {resume.get('location', 'N/A')}", unsafe_allow_html=True)

    # Education
    if 'education' in resume:
        st.markdown("### 🎓 Education")
        if isinstance(resume['education'], list):
            for edu in resume['education']:
                if isinstance(edu, dict):
                    st.markdown(f"**{edu.get('degree', 'Degree')}** - {edu.get('institution', 'Institution')}", unsafe_allow_html=True)
                    st.markdown(f"{edu.get('start_date', '')} - {edu.get('end_date', '')} | {edu.get('location', '')}", unsafe_allow_html=True)
                else:
                    st.markdown(f"- {edu}", unsafe_allow_html=True)
        else:
            st.markdown(f"- {resume['education']}", unsafe_allow_html=True)

    # Experience
    if 'experience' in resume:
        st.markdown("### 💼 Experience")
        if isinstance(resume['experience'], list):
            for exp in resume['experience']:
                if isinstance(exp, dict):
                    st.markdown(f"**{exp.get('title', 'Role')}** at {exp.get('company', 'Company')}", unsafe_allow_html=True)
                    st.markdown(f"{exp.get('start_date', '')} - {exp.get('end_date', '')} | {exp.get('location', '')}", unsafe_allow_html=True)
                    st.markdown(f"{exp.get('description', '')}", unsafe_allow_html=True)
                else:
                    st.markdown(f"- {exp}", unsafe_allow_html=True)
        else:
            st.markdown(f"- {resume['experience']}", unsafe_allow_html=True)

    # Projects
    if 'projects' in resume:
        st.markdown("### 🛠️ Projects")
        if isinstance(resume['projects'], list):
            for proj in resume['projects']:
                if isinstance(proj, dict):
                    st.markdown(f"**{proj.get('title', 'Project')}**", unsafe_allow_html=True)
                    st.markdown(f"{proj.get('description', '')}", unsafe_allow_html=True)
                else:
                    st.markdown(f"- {proj}", unsafe_allow_html=True)
        else:
            st.markdown(f"- {resume['projects']}", unsafe_allow_html=True)

    # Certifications
    if 'certifications' in resume:
        st.markdown("### 📜 Certifications")
        if isinstance(resume['certifications'], list):
            for cert in resume['certifications']:
                if isinstance(cert, dict):
                    st.markdown(f"**{cert.get('title', 'Certification')}** - {cert.get('issuer', '')} ({cert.get('year', '')})", unsafe_allow_html=True)
                    if 'link' in cert:
                        st.markdown(f"[🔗 View Certificate]({cert['link']})")
                else:
                    st.markdown(f"- {cert}", unsafe_allow_html=True)
        else:
            st.markdown(f"- {resume['certifications']}", unsafe_allow_html=True)

    # Languages
    if 'languages' in resume and resume['languages']:
        st.markdown("### 🌍 Languages")
        if isinstance(resume['languages'], list):
            st.markdown(", ".join(resume['languages']), unsafe_allow_html=True)
        else:
            st.markdown(resume['languages'], unsafe_allow_html=True)

    # Skills
    if 'skills' in resume and resume['skills']:
        st.markdown("### 🏅 Skills")
        if isinstance(resume['skills'], list):
            st.markdown(", ".join(resume['skills']), unsafe_allow_html=True)
        else:
            st.markdown(resume['skills'], unsafe_allow_html=True)

    # Social Profiles
    if 'social_profiles' in resume and resume['social_profiles']:
        st.markdown("### 🌐 Social Profiles")
        if isinstance(resume['social_profiles'], list):
            for profile in resume['social_profiles']:
                if isinstance(profile, dict):
                    st.markdown(f"[{profile.get('platform', 'Profile')}]({profile.get('link', '')})", unsafe_allow_html=True)
                else:
                    st.markdown(f"- {profile}", unsafe_allow_html=True)
        else:
            st.markdown(resume['social_profiles'], unsafe_allow_html=True)

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="HR Bot Resume Search", 
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
    .result-count {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for search controls
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/find-matching-job.png", width=80)
        st.title("HR Bot Resume Search")
        
        st.markdown("### Search")
        search_query = st.text_input("Enter your search query:", placeholder="e.g., Python AND MachineLearning")
        
        with st.expander("Search Tips"):
            st.markdown("""
            - **Simple keyword**: `Python`
            - **AND operator**: `Python AND Django`
            - **OR operator**: `JavaScript OR TypeScript`
            - **Grouped logic**: `(Python OR Java) AND (AWS OR Azure)`
            - **Multi-word skills**: `MachineLearning` or `HuggingFace`
            """)
        
        st.divider()
        st.markdown("""
        **About**  
        Find relevant candidates by matching keywords and phrases in their profiles. Supports Boolean search for precise filtering.
        """)

    # Main content
    st.title("🔎 Looking for some candidates?")
    
    if not search_query:
        st.info("👈 Enter a search query in the sidebar to begin searching.")
        
        # Sample placeholders when no search is performed
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🚀 Features
            - **Boolean Logic**: Complex search queries
            - **Fast Search**: Optimized algorithm
            - **Detailed View**: See complete candidate profiles
            - **User-friendly**: Intuitive interface
            """)
        with col2:
            st.markdown("""
            ### 💡 Example Queries
            - `Python AND (Django OR Flask)`
            - `JavaScript AND React`
            - `AWS OR Azure`
            """)
        return

    # Parse Boolean Query
    bsp = BooleanSearchParser()
    if 'AND' in search_query or 'OR' in search_query or 'NOT' in search_query or '"' in search_query:
        try:
            parsed_query = bsp.parse_query(search_query)
        except Exception as e:
            st.error(f"❌ Error parsing query: {e}")
            return
    else:
        parsed_query = Symbol(search_query.lower())

    # Connect to MongoDB
    try:
        with st.spinner("Connecting to database..."):
            client = MongoClient(config.MONGO_URI)
            coll = client[config.DB_NAME][config.COLLECTION_NAME]
            docs = list(coll.find({}))
            st.success(f"📁 Loaded {len(docs)} resumes from database")
    except Exception as e:
        st.error(f"❌ Failed to load resumes: {e}")
        return

    # Search resumes
    st.subheader("🔍 Searching resumes...")
    progress_bar = st.progress(0)
    
    # Store unique documents using a dictionary with _id as key
    unique_matching_docs = {}
    
    for idx, doc in enumerate(docs):
        try:
            doc_id = str(doc.get('_id'))
            
            # Skip if we've already processed this document
            if doc_id in unique_matching_docs:
                continue
                
            raw_text = flatten_json(doc)
            norm_text = normalize(raw_text)
            
            # Debug output for search terms
            if st.session_state.get('debug_search', False):
                st.write(f"Searching in document {doc_id}:")
                st.write(f"Normalized text: {norm_text[:200]}...")
            
            if evaluate_expression(parsed_query, norm_text, bsp.quoted_phrases):
                unique_matching_docs[doc_id] = doc
                # Extract search terms for highlighting
                search_terms = extract_search_terms(parsed_query, bsp.quoted_phrases)
                st.session_state[f"matched_terms_{doc_id}"] = search_terms
                
                # Pre-highlight the entire document
                highlighted_doc = highlight_dict_values(doc, search_terms)
                st.session_state[f"highlighted_doc_{doc_id}"] = highlighted_doc
        except Exception as e:
            st.warning(f"⚠️ Error processing document {doc.get('_id')}: {e}")
        progress_bar.progress((idx + 1) / len(docs))

    progress_bar.empty()

    # Get list of unique matching documents
    matching_docs_list = list(unique_matching_docs.values())

    # Display results
    if matching_docs_list:
        st.markdown(f"<div class='result-count'>✅ Found {len(matching_docs_list)} matching candidates</div>", unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Card View", "Table View"])
        
        with tab1:
            # Card view
            for doc in matching_docs_list:
                doc_id = str(doc.get('_id'))
                matched_terms = st.session_state.get(f"matched_terms_{doc_id}", set())
                
                with st.container():
                    # Highlight the name and contact info
                    name = highlight_text(doc.get('name', 'Unknown Candidate'), matched_terms)
                    email = highlight_text(doc.get('email', 'No email provided'), matched_terms)
                    phone = highlight_text(doc.get('phone', 'No phone provided'), matched_terms)
                    
                    st.markdown(f"""
                    <div class="card">
                        <div class="candidate-name">{name}</div>
                        <div class="contact-info">
                            📧 {email} | 
                            📱 {phone}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([4, 1])
                    
                    # Extract top skills/keywords for preview
                    skills = doc.get('skills', [])
                    if skills:
                        if isinstance(skills, list):
                            skill_text = ", ".join(skills[:5])
                            if len(skills) > 5:
                                skill_text += "..."
                        else:
                            skill_text = str(skills)
                        highlighted_skills = highlight_text(skill_text, matched_terms)
                        col1.markdown(f"**Skills**: {highlighted_skills}", unsafe_allow_html=True)
                    
                    # View details button with unique key
                    button_key = f"view_{doc.get('_id')}"
                    if col2.button("View Details", key=button_key):
                        st.session_state[f"show_details_{doc.get('_id')}"] = True
                    
                    # Show details if button was clicked
                    if st.session_state.get(f"show_details_{doc.get('_id')}", False):
                        with st.expander("📄 Full Resume Details", expanded=True):
                            tabs = st.tabs(["Formatted View", "JSON View"])
                            
                            with tabs[0]:
                                # Formatted structured view
                                doc_id = str(doc.get('_id'))
                                matched_terms = st.session_state.get(f"matched_terms_{doc_id}", set())
                                highlighted_doc = st.session_state.get(f"highlighted_doc_{doc_id}", doc)
                                
                                render_formatted_resume(highlighted_doc)
                            
                            with tabs[1]:
                                # Raw JSON view with pretty formatting
                                display_json(doc)
        
        with tab2:
            # Table view for comparison
            table_data = []
            for doc in matching_docs_list:
                row = {
                    "Name": doc.get('name', 'Unknown'),
                    "Email": doc.get('email', 'N/A'),
                    "Phone": doc.get('phone', 'N/A'),
                    "Location": doc.get('location', 'N/A'),
                }
                
                # Add skills as comma-separated string
                skills = doc.get('skills', [])
                if isinstance(skills, list):
                    row["Skills"] = ", ".join(skills[:3]) + ("..." if len(skills) > 3 else "")
                else:
                    row["Skills"] = str(skills)
                
                table_data.append(row)
            
            st.dataframe(table_data, use_container_width=True)
    else:
        st.info("🔎 No resumes matched your search query. Try adjusting your terms.")
        st.markdown("""
        **Tips to improve results:**
        - Use broader terms
        - Try using OR instead of AND
        - Check for typos in your search query
        - Simplify complex Boolean expressions
        """)

def run_retriever():
    if 'init' not in st.session_state:
        st.session_state.init = True
        # Initialize any other session state variables here
    
    main()

__all__ = ['run_retriever', 'render_formatted_resume']
