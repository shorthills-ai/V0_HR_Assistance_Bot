import json
import re
from typing import List, Dict, Set, Tuple
from pymongo import MongoClient
import streamlit as st
import config
import openai
from openai import AzureOpenAI
from bson.objectid import ObjectId

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
        system_prompt = """You are an expert at writing detailed, professional project descriptions for resumes.
Your task is to expand a minimal project description into a comprehensive, well-structured description that:
1. Focuses STRICTLY on aspects related to the provided job keywords
2. Maintains the original intent and scope while emphasizing keyword-relevant details
3. Includes ONLY technologies and tools that match or are directly related to the job keywords
4. Describes impact and outcomes in the context of the job requirements
5. Uses professional, formal language
6. Is based ONLY on the information provided - do not invent or hallucinate
7. MUST be at least 150 words long (approximately 8-10 sentences)
8. Each sentence should connect to at least one job keyword

Important:
- Do NOT invent technologies, tools, or outcomes
- Do NOT add specific metrics unless provided
- Keep the description factual and plausible
- Use only the information from the project title and minimal description
- Focus EXCLUSIVELY on aspects that align with the job keywords
- Structure the description with:
  * First 2-3 sentences: Project overview, purpose, and scope
  * Middle 4-5 sentences: Detailed technical implementation, methodologies, and challenges
  * Final 2-3 sentences: Impact, outcomes, and business value
- If the project doesn't match any job keywords, return the original description unchanged
- Do NOT add new projects or features that weren't in the original description"""

        user_prompt = f"""Expand this project description into a detailed, professional description that focuses STRICTLY on the job keywords.

Project Title: {project.get('title', '')}
Original Description: {project.get('description', '')}
Job Keywords: {', '.join(job_keywords)}

Write a detailed description (at least 150 words, approximately 8-10 sentences) that:
1. Focuses EXCLUSIVELY on aspects related to the job keywords
2. Maintains the original scope while emphasizing keyword-relevant details
3. Includes ONLY technologies and tools that match the job keywords
4. Uses professional language
5. Is based only on the provided information
6. Follows this structure:
   - First 2-3 sentences: Project overview, purpose, and scope
   - Middle 4-5 sentences: Detailed technical implementation, methodologies, and challenges
   - Final 2-3 sentences: Impact, outcomes, and business value

Important: 
- The final description MUST be at least 150 words long
- Each sentence should connect to at least one job keyword
- Do not include details unrelated to the job keywords
- Focus on technical depth and implementation details
- If the project doesn't match any job keywords, return the original description unchanged
- Do NOT add new projects or features that weren't in the original description"""

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
            
            # Verify word count and keyword relevance
            word_count = len(expanded_description.split())
            if word_count < 150:
                # If too short, ask for more details focusing on keywords
                follow_up_prompt = f"""The expanded description is too short ({word_count} words) and needs to focus more on the job keywords. Please expand it to at least 150 words by adding more technical details and implementation specifics that are directly related to these job keywords: {', '.join(job_keywords)}

Current Description:
{expanded_description}

Please provide a more detailed version that:
1. Is at least 150 words long
2. Focuses STRICTLY on the job keywords
3. Includes more technical details related to the keywords
4. Maintains accuracy and relevance
5. Follows the structure:
   - First 2-3 sentences: Project overview, purpose, and scope
   - Middle 4-5 sentences: Detailed technical implementation, methodologies, and challenges
   - Final 2-3 sentences: Impact, outcomes, and business value
6. Does NOT add any new features or technologies not mentioned in the original description"""
                
                follow_up_response = self.client.chat.completions.create(
                    model=st.secrets["azure_openai"]["deployment"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": follow_up_prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "text"}
                )
                expanded_description = follow_up_response.choices[0].message.content.strip()
            
            # Verify keyword relevance and structure
            keyword_check_prompt = f"""Verify that this project description focuses on the job keywords, is at least 150 words long, and follows the required structure.

Job Keywords: {', '.join(job_keywords)}
Current Description: {expanded_description}

If the description:
1. Is not at least 150 words long, or
2. Does not focus enough on the job keywords, or
3. Does not follow the required structure:
   - First 2-3 sentences: Project overview, purpose, and scope
   - Middle 4-5 sentences: Detailed technical implementation, methodologies, and challenges
   - Final 2-3 sentences: Impact, outcomes, and business value
4. Contains any new features or technologies not in the original description

Please provide an improved version that addresses these issues."""
            
            final_check_response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": keyword_check_prompt}
                ],
                temperature=0.3,
                response_format={"type": "text"}
            )
            
            final_description = final_check_response.choices[0].message.content.strip()
            
            # Final verification of word count
            if len(final_description.split()) < 150:
                return expanded_description  # Return the longer version if final check made it too short
            
            return final_description
            
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

    def retailor_resume(self, original_resume: Dict, job_keywords: Set[str], job_description: str = "") -> Dict:
        """Retailor the resume to only keep skills and projects that match the job keywords, and rewrite the summary to be job-specific."""
        safe_resume = convert_objectid_to_str(original_resume)
        
        # Generate job-specific title
        if job_description:
            safe_resume["title"] = self.generate_job_specific_title(safe_resume, job_keywords, job_description)
        
        # Expand short project descriptions
        if "projects" in safe_resume:
            for project in safe_resume["projects"]:
                if len(project.get("description", "").split()) < 60:  # If description is less than 60 words
                    project["description"] = self.expand_project_description(project, job_keywords)
        
        system_prompt = """
You are an AI assistant that retailors resumes to match a job description.
Your ONLY task is to:
- Filter the skills and projects sections of the resume so that they include ONLY those that directly match the provided job keywords.
- Include ONLY those skills that directly match the job keywords (case-insensitive, substring match allowed). Do NOT include unrelated or generic skills. Do NOT hallucinate or invent skills.
- Rewrite the 'summary' field to be a concise, job-specific summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords.
- Add or update a 'title' field in the resume JSON, inferring a proper, professional job title for the candidate based on the job description and their experience/skills. The title should be a realistic job title (e.g., 'Frontend Developer', 'Data Scientist', 'Project Manager'), not just a single keyword or technology. Do not leave the title blank. If unsure, use the most relevant job title from the job description.
- In addition to the 'projects' section, also review the 'experience' section. If any work experience description contains relevant projects or achievements that match the job keywords, extract those as additional projects or ensure they are included in the retailored resume's projects section.
- Do NOT add, invent, or hallucinate any new skills, projects, or summary content.
- Do NOT change or add any other fields except 'title', 'skills', 'projects', and 'summary'.
- The output must be the same JSON structure as the input, but with skills, projects, summary, and title updated as above.
- If no skills or projects match, leave those sections empty.
- Do not change the order or content of any other fields.
"""
        user_prompt = f"""
Job Keywords:
{json.dumps(list(job_keywords))}

Original Resume:
{json.dumps(safe_resume, indent=2)}

Instructions:
1. Filter the 'skills' list to include ONLY those that match the job keywords (case-insensitive, substring match allowed).
2. Filter the 'projects' list to include ONLY those that mention any of the job keywords in their description or technologies.
3. Rewrite the 'summary' field to be a 2-4 sentence summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords.
4. Do NOT add or invent any new skills, projects, or summary content.
5. Return the complete resume in the exact same JSON format, with only the filtered skills, projects, and new summary.
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