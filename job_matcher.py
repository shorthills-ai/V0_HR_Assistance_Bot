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
    
    def retailor_resume(self, original_resume: Dict, job_keywords: Set[str]) -> Dict:
        """Retailor the resume to only keep skills and projects that match the job keywords, and rewrite the summary to be job-specific."""
        safe_resume = convert_objectid_to_str(original_resume)
        system_prompt = """
You are an AI assistant that retailors resumes to match a job description.
Your ONLY task is to:
- Filter the skills and projects sections of the resume so that they include ONLY those that directly match the provided job keywords.
- Rewrite the 'summary' field to be a concise, job-specific summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords.
- Do NOT add, invent, or hallucinate any new skills, projects, or summary content.
- Do NOT change or add any other fields.
- The output must be the same JSON structure as the input, but with skills, projects, and summary updated as above.
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
            candidates = list(self.collection.find(query).limit(10))
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