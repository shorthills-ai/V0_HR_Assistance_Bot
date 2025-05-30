import json
import re
from typing import List, Dict, Set, Tuple
from pymongo import MongoClient
import streamlit as st
import config
import openai
from openai import AzureOpenAI
from bson.objectid import ObjectId
import time

class JobDescriptionAnalyzer:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=st.secrets["azure_openai"]["api_key"],
            api_version=st.secrets["azure_openai"]["api_version"],
            azure_endpoint=st.secrets["azure_openai"]["endpoint"]
        )
        
    def extract_keywords(self, job_description: str) -> Dict[str, Set[str]]:
        """Extract keywords from job description using Azure OpenAI."""
        system_prompt = """You are an AI assistant that extracts ONLY the most relevant and specific keywords from job descriptions. 
Focus on extracting:
1. Required technical skills and technologies
2. Programming languages
3. Tools and frameworks
4. Key responsibilities that indicate required skills

Important rules:
- Extract ONLY keywords that are explicitly mentioned in the job description
- DO NOT add any keywords that are not directly stated
- DO NOT infer or assume additional skills
- Return a JSON object with a single array field named 'keywords'
- Keep the list focused and specific
- Avoid generic terms unless explicitly required"""

        user_prompt = f"""Extract ONLY the explicitly mentioned technical keywords from this job description. Return a JSON object with a 'keywords' array.

Job Description:
{job_description}"""
        
        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature for more focused extraction
                response_format={ "type": "json_object" }
            )
            
            # Parse the response to get keywords
            keywords_text = response.choices[0].message.content.strip()
            response_data = json.loads(keywords_text)
            
            if not isinstance(response_data, dict) or 'keywords' not in response_data:
                st.error("Invalid response format from keyword extraction")
                return {"keywords": set(), "technologies": set()}
            
            keywords = set(response_data['keywords'])
            
            return {
                "keywords": keywords,
                "technologies": keywords
            }
        except Exception as e:
            st.error(f"Error extracting keywords: {str(e)}")
            return {
                "keywords": set(),
                "technologies": set()
            }

class CandidateScorer:
    def __init__(self, job_keywords: Dict[str, Set[str]]):
        self.job_keywords = job_keywords
        self.client = AzureOpenAI(
            api_key=st.secrets["azure_openai"]["api_key"],
            api_version=st.secrets["azure_openai"]["api_version"],
            azure_endpoint=st.secrets["azure_openai"]["endpoint"]
        )
        
    def calculate_score(self, candidate: Dict) -> Tuple[int, str]:
        """Calculate a score for the candidate using Azure OpenAI."""
        # Prepare the evaluation data
        evaluation_data = {
            "job_description": list(self.job_keywords["keywords"]),
            "candidate": {
                "id": str(candidate.get("_id", "N/A")),
                "name": candidate.get("name", "Unknown"),
                "phone": candidate.get("phone", "N/A"),
                "email": candidate.get("email", "N/A"),
                "skills": candidate.get("skills", []),
                "projects": candidate.get("projects", [])
            }
        }
        
        system_prompt = """You are an AI designed to evaluate candidate suitability for a job based on pre-extracted job description keywords.
Your task is to compare the candidate's skills and projects against the job description keywords and assign a holistic match score.

Evaluation Guidelines:
1. Primary Focus (80% of score):
   - Skills match with job requirements
   - Project relevance and implementation of required technologies
   
2. Secondary Focus (20% of score):
   - Education relevance
   - Experience relevance
   - Certifications

Scoring Rules:
- Score range: 1-100
- Focus on exact matches and closely related technologies
- Higher scores for candidates with multiple, highly relevant matches
- Lower scores for partial matches or minimal alignment
- Do not assume or hallucinate missing information
- Explicitly mention missing required skills in the reason

Status Rules:
- "Accepted" if score > 70
- "Rejected" if score ≤ 70"""

        user_prompt = f"""Evaluate the candidate's suitability based on the job description keywords. Focus primarily on skills and projects.

### Job Description Keywords:
{json.dumps(evaluation_data["job_description"])}

### Candidate Details:
- Name: {evaluation_data["candidate"]["name"]}
- Skills: {json.dumps(evaluation_data["candidate"]["skills"])}
- Projects: {json.dumps(evaluation_data["candidate"]["projects"])}

### Required Output Format (JSON):
{{
  "mongo_id": "{evaluation_data["candidate"]["id"]}",
  "name": "{evaluation_data["candidate"]["name"]}",
  "phone": "{evaluation_data["candidate"]["phone"]}",
  "email": "{evaluation_data["candidate"]["email"]}",
  "score": <number between 1 and 100>,
  "reason": "<detailed explanation focusing on skills and projects matches>",
  "status": "<'Accepted' if score > 70, 'Rejected' if score <= 70>"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={ "type": "json_object" }
            )
            
            # Get the response text and clean it
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                result = json.loads(response_text)
                
                # Validate required fields
                required_fields = ["score", "reason", "status"]
                if not all(field in result for field in required_fields):
                    raise ValueError("Missing required fields in response")
                
                # Ensure score is a number
                if not isinstance(result["score"], (int, float)):
                    result["score"] = int(result["score"])
                
                # Ensure status is valid
                if result["status"] not in ["Accepted", "Rejected"]:
                    result["status"] = "Rejected" if result["score"] <= 70 else "Accepted"
                
                return result["score"], result["reason"]
                
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON response from LLM: {str(e)}")
                return 0, "Error: Invalid response format from evaluation system"
            
        except Exception as e:
            st.error(f"Error evaluating candidate: {str(e)}")
            return 0, f"Error during evaluation: {str(e)}"

def convert_objectid_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_objectid_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(i) for i in obj]
    elif type(obj).__name__ == "ObjectId":
        return str(obj)
    else:
        return obj

class ResumeRetailor:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=st.secrets["azure_openai"]["api_key"],
            api_version=st.secrets["azure_openai"]["api_version"],
            azure_endpoint=st.secrets["azure_openai"]["endpoint"]
        )
    
    def expand_project_description(self, project: Dict, job_keywords: Set[str]) -> str:
        """Expand a project description based on its title and minimal description."""
        # Only use the main system and user prompt for every project, no fallback or legacy prompt logic
        system_prompt = """
You are a professional technical resume writer. Your task is to convert raw project data into a concise, impactful, and metric-driven project description suitable for a resume.

Follow these rules:

1. Output 4 to 7 bullet points, each 1–2 lines long maximum. Dont write extra, be to the point and concise.
2. The first line compulsorily has to be a professionally retailored title of the project, do this for every project and make sure it has only first letter of each word capitalised.
    -> If its a work experience,make sure not to include company name in the title and give a generic title to the specific project.
3. Each bullet must start with a strong past-tense action verb (Engineered, Built, Automated, Designed, Achieved, etc.).
4. The bullet points should 
 -> summarize what you built, what technologies you used, and why it mattered — no fluff, no LLM mentions unless critical.
 -> describe any UX, responsiveness, optimization, version control, or frontend/backend implementation details.
 -> include specific results using clear metrics — e.g., "93% coverage," "reduced latency by 40%," "automated 80% of manual work."
 -> explain the business or user impact — how it helped the client, user, or team.
5. Keep each bullet short and crisp — aim for clarity, not verbosity.
6. Use plain text only — no markdown, no bullet symbols.
7. IMPORTANT: Make sure to highlight and emphasize any skills, technologies, or experiences that match the provided job keywords.
8. If the project uses any of the job keywords, make sure to explicitly mention them in the description.

You must NOT generate a paragraph — return 4 to 7 standalone resume-ready bullets.
"""

        user_prompt = f"""
Project Title: {project['title']}
Raw Description: {project['description']}
Technologies: {', '.join(project.get('technologies', []))}
Job Keywords to Highlight: {', '.join(job_keywords)}

Using the system instructions, create 4 to 7 clear, action-driven bullet points that summarize the project. Make sure to emphasize any matches with the job keywords in the description.
"""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "text"}
            )
            
            expanded_description = response.choices[0].message.content.strip()
            
            return expanded_description
            
        except Exception as e:
            st.error(f"Error expanding project description: {str(e)}")
            return project.get('description', '')
    
    def generate_job_specific_title(self, candidate: Dict, job_keywords: Set[str], job_description: str) -> str:
        """Generate a job-specific title for the candidate based on their profile and job description."""
        system_prompt = """You are an expert HR professional specializing in job title creation and matching.
Your task is to create a specific, professional job title that:
1. Accurately reflects the candidate's experience and skills
2. Aligns with the job description requirements
3. Uses industry-standard job titles
4. Is specific enough to be meaningful but not overly narrow
5. Matches the seniority level implied by the candidate's experience

Important rules:
- Use ONLY information from the candidate's profile and job description
- Do NOT invent or assume experience or skills
- Choose from standard industry job titles
- Consider the candidate's actual experience level
- Match the job description's requirements
- Be specific to the role and industry
- Do NOT use generic titles like 'Professional' or 'Specialist' unless specifically required
- Do NOT use titles that overstate the candidate's experience
- Return ONLY the job title, no explanation or additional text"""

        user_prompt = f"""Create a specific, professional job title for this candidate based on their profile and the job description.

Candidate Profile:
- Name: {candidate.get('name', '')}
- Current Title: {candidate.get('title', '')}
- Skills: {', '.join(candidate.get('skills', []))}
- Projects: {json.dumps(candidate.get('projects', []), indent=2)}
- Education: {json.dumps(candidate.get('education', []), indent=2)}

Job Description Keywords: {', '.join(job_keywords)}

Job Description:
{job_description}

Create a specific job title that:
1. Matches the candidate's actual experience and skills
2. Aligns with the job requirements
3. Uses industry-standard terminology
4. Reflects appropriate seniority level
5. Is specific to the role and industry

Return ONLY the job title, no additional text."""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "text"}
            )
            
            title = response.choices[0].message.content.strip()
            
            # Verify the title is appropriate
            verification_prompt = f"""Verify that this job title is appropriate for the candidate and job description.

Proposed Title: {title}

Candidate Profile:
- Name: {candidate.get('name', '')}
- Current Title: {candidate.get('title', '')}
- Skills: {', '.join(candidate.get('skills', []))}

Job Keywords: {', '.join(job_keywords)}

If the title:
1. Overstates the candidate's experience
2. Is too generic
3. Doesn't match the job requirements
4. Uses non-standard terminology

Please provide a more appropriate title. Otherwise, return the same title."""

            verification_response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.3,
                response_format={"type": "text"}
            )
            
            final_title = verification_response.choices[0].message.content.strip()
            return final_title
            
        except Exception as e:
            st.error(f"Error generating job title: {str(e)}")
            return candidate.get('title', '')  # Return original title if there's an error

    def extract_relevant_projects(self, resume: Dict, job_keywords: Set[str]) -> list:
        """Extract relevant projects from both 'projects' and 'experience' sections based on job keywords."""
        keywords_lower = {k.lower() for k in job_keywords}
        relevant_projects = []
        # Check original projects
        for proj in resume.get('projects', []):
            desc = proj.get('description', '').lower()
            techs = [t.lower() for t in proj.get('technologies', [])] if 'technologies' in proj else []
            if any(k in desc for k in keywords_lower) or any(k in techs for k in keywords_lower):
                relevant_projects.append(proj)
        # Check experience descriptions
        for exp in resume.get('experience', []):
            exp_desc = exp.get('description', '').lower()
            if any(k in exp_desc for k in keywords_lower):
                # Create a project-like dict from experience
                relevant_projects.append({
                    'title': exp.get('title', 'Relevant Experience Project'),
                    'description': exp.get('description', ''),
                    'technologies': exp.get('technologies', []) if 'technologies' in exp else []
                })
        return relevant_projects

    def retailor_resume(self, original_resume: Dict, job_keywords: Set[str], job_description: str = "") -> Dict:
        """Retailor the resume to keep all original skills unchanged and only modify projects based on job relevance."""
        safe_resume = convert_objectid_to_str(original_resume)
        
        # Generate job-specific title
        if job_description:
            safe_resume["title"] = self.generate_job_specific_title(safe_resume, job_keywords, job_description)
        
        # Use helper to get all relevant projects (from both projects and experience)
        relevant_projects = self.extract_relevant_projects(safe_resume, job_keywords)
        
        # --- Retailor the resume (keeping all skills, only modifying projects) ---
        system_prompt = """
You are an AI assistant that retailors resumes to match a job description.
Your ONLY task is to:
- KEEP ALL ORIGINAL SKILLS UNCHANGED - Do NOT filter or remove any skills from the candidate's original skill set
- Filter the projects section to include ONLY those projects that directly match or relate to the provided job keywords
- Rewrite the 'summary' field to be a concise, job-specific summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords
- Add or update a 'title' field in the resume JSON, inferring a proper, professional job title for the candidate based on the job description and their experience/skills. The title should be a realistic job title (e.g., 'Frontend Developer', 'Data Scientist', 'Project Manager'), not just a single keyword or technology. Do not leave the title blank. If unsure, use the most relevant job title from the job description.
- In addition to the 'projects' section, also review the 'experience' section. If any work experience description contains relevant projects or achievements that match the job keywords, extract those as additional projects or ensure they are included in the retailored resume's projects section.
- Do NOT add, invent, or hallucinate any new skills, projects, or summary content.
- Do NOT change or add any other fields except 'title', 'projects', and 'summary'. KEEP 'skills' exactly as provided in the original resume.
- The output must be the same JSON structure as the input, but with projects, summary, and title updated as above.
- If no projects match, leave the projects section empty.
- Do not change the order or content of any other fields.
"""
        user_prompt = f"""
Job Keywords:
{json.dumps(list(job_keywords))}

Original Resume:
{json.dumps(safe_resume, indent=2)}

Instructions:
1. KEEP ALL ORIGINAL SKILLS UNCHANGED - Copy the entire 'skills' list exactly as provided in the original resume.
2. Filter the 'projects' list to include ONLY those that mention any of the job keywords in their description or technologies.
3. Rewrite the 'summary' field to be a 2-4 sentence summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords.
4. Do NOT add or invent any new skills, projects, or summary content.
5. Return the complete resume in the exact same JSON format, with the original skills kept unchanged, filtered projects, and new summary.
"""
        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={ "type": "json_object" }
            )
            # Parse the response
            retailored_resume = json.loads(response.choices[0].message.content.strip())
            
            # IMPORTANT: Ensure all original skills are preserved and add any missing job keywords
            original_skills = set(safe_resume.get("skills", []))
            # Convert job keywords to lowercase for case-insensitive comparison
            job_keywords_lower = {k.lower() for k in job_keywords}
            original_skills_lower = {s.lower() for s in original_skills}
            
            # Add any job keywords that aren't already in the skills list
            new_skills = []
            for keyword in job_keywords:
                if keyword.lower() not in original_skills_lower:
                    new_skills.append(keyword)
            
            # Combine original skills with new skills
            retailored_resume["skills"] = list(original_skills) + new_skills
            
            # --- Custom logic: Only enhance all projects if there is <= 1 relevant project, else keep only relevant projects ---
            if "projects" in retailored_resume:
                filtered_projects = retailored_resume["projects"]
                original_projects = safe_resume.get("projects", [])
                if len(relevant_projects) == 0:
                    # If no relevant projects found, include all original projects
                    filtered_projects = original_projects
                else:
                    # If 1 or more relevant projects found, keep only those relevant projects
                    filtered_projects = relevant_projects
                for project in filtered_projects:
                    project["description"] = self.expand_project_description(project, job_keywords)
                    time.sleep(1)
                retailored_resume["projects"] = filtered_projects
            
            # Validate the structure
            if not self._validate_resume_structure(safe_resume, retailored_resume):
                st.error("Error: Retailored resume structure doesn't match original")
                return safe_resume
            return retailored_resume
        except Exception as e:
            st.error(f"Error retailoring resume: {str(e)}")
            return safe_resume
    
    def _validate_resume_structure(self, original: Dict, retailored: Dict) -> bool:
        """Validate that the retailored resume maintains the original structure."""
        # Check if all original fields are present
        for key in original.keys():
            if key not in retailored:
                return False
        
        # Check if no new fields were added
        for key in retailored.keys():
            if key not in original:
                return False
                
        return True

class JobMatcher:
    def __init__(self):
        self.client = MongoClient(config.MONGO_URI)
        self.db = self.client[config.DB_NAME]
        self.collection = self.db[config.COLLECTION_NAME]
        self.resume_retailor = ResumeRetailor()
        
    def pre_filter_candidates(self, keywords: Set[str]) -> List[Dict]:
        """Pre-filter candidates based on skills and projects."""
        if not keywords:
            st.warning("No keywords extracted from job description")
            return []
            
        # Convert keywords to lowercase for case-insensitive matching
        keywords_lower = [k.lower() for k in keywords]
        
        # Build the MongoDB query
        query = {
            "$or": [
                # Match in skills array
                {"skills": {"$in": keywords_lower}},
                # Match in projects description
                {"projects.description": {"$regex": "|".join(keywords_lower), "$options": "i"}},
                # Match in projects technologies
                {"projects.technologies": {"$in": keywords_lower}}
            ]
        }
        
        try:
            # Execute the query with a limit
            candidates = list(self.collection.find(query))
            if not candidates:
                st.info("No candidates found matching the keywords")
            return candidates
        except Exception as e:
            st.error(f"Error querying database: {str(e)}")
            return []
        
    def find_matching_candidates(self, job_description: str, progress_bar=None, status_text=None) -> List[Dict]:
        """Find and score candidates matching the job description."""
        if not job_description.strip():
            st.error("Please provide a job description")
            return []
            
        # Extract keywords from job description
        analyzer = JobDescriptionAnalyzer()
        keywords = analyzer.extract_keywords(job_description)
        
        
        # Pre-filter candidates based on keywords
        candidates = self.pre_filter_candidates(keywords["keywords"])
        
        if not candidates:
            return []
        
        # Score each pre-filtered candidate
        scored_candidates = []
        total_candidates = len(candidates)
        
        for idx, candidate in enumerate(candidates):
            # Update progress
            if progress_bar and status_text:
                progress = (idx + 1) / total_candidates
                progress_bar.progress(progress)
                status_text.text(f"Evaluating candidate {idx + 1} of {total_candidates}")
            
            try:
                scorer = CandidateScorer(keywords)
                score, reason = scorer.calculate_score(candidate)
                
                # Only include candidates with score > 0
                if score > 0:
                    scored_candidates.append({
                        "mongo_id": str(candidate.get("_id")),
                        "name": candidate.get("name", "Unknown"),
                        "phone": candidate.get("phone", "N/A"),
                        "email": candidate.get("email", "N/A"),
                        "score": score,
                        "reason": reason,
                        "status": "Accepted" if score > 70 else "Rejected",
                        "resume": candidate
                    })
            except Exception as e:
                st.error(f"Error evaluating candidate {candidate.get('name', 'Unknown')}: {str(e)}")
                continue
        
        # Sort by score in descending order
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        if not scored_candidates:
            st.warning("No candidates met the minimum score threshold")
            
        return scored_candidates 

    def retailor_candidate_resume(self, candidate_id: str, job_keywords: Set[str]) -> Dict:
        """Retailor a specific candidate's resume."""
        try:
            # Find the candidate
            candidate = self.collection.find_one({"_id": ObjectId(candidate_id)})
            if not candidate:
                st.error("Candidate not found")
                return None
                
            # Retailor the resume
            retailored_resume = self.resume_retailor.retailor_resume(candidate, job_keywords)
            
            return retailored_resume
            
        except Exception as e:
            st.error(f"Error retailoring resume: {str(e)}")
            return None 