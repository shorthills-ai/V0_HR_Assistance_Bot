import streamlit as st
import json
import re
import time
import random
from typing import Dict, List, Set, Tuple
from pymongo import MongoClient
from bson import ObjectId
from openai import AzureOpenAI
import config

class JobDescriptionAnalyzer:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=st.secrets["azure_openai"]["api_key"],
            api_version=st.secrets["azure_openai"]["api_version"],
            azure_endpoint=st.secrets["azure_openai"]["endpoint"]
        )
        
    def extract_keywords(self, job_description: str) -> Dict[str, Set[str]]:
        """Extract keywords from job description using Azure OpenAI with enhanced parsing of parenthetical content."""
        system_prompt = """You are an AI assistant that intelligently extracts technical keywords from job descriptions. 
You MUST parse content within parentheses, brackets, and comma-separated lists to extract ALL individual keywords.

CRITICAL PARSING RULES:

1. **Parentheses/Brackets Parsing**: For content like "Large Language Models (LLMs)", extract BOTH:
   - The main term: "Large Language Models"
   - The content inside: "LLMs"

2. **Comma-Separated Lists**: For content like "Fine-tuning (LoRA, PEFT, QLoRA, etc.)", extract:
   - The main term: "Fine-tuning"
   - Each individual item: "LoRA", "PEFT", "QLoRA"
   - IGNORE: "etc.", "and more", "among others"

3. **Multiple Items in Parentheses**: For "Transformers (HuggingFace Transformers, BERT, GPT, LLaMA, etc.)", extract:
   - "Transformers"
   - "HuggingFace Transformers" 
   - "BERT"
   - "GPT"
   - "LLaMA"

4. **Nested Technologies**: For "Vector Databases (Pinecone, Weaviate, Qdrant, Chroma)", extract:
   - "Vector Databases"
   - "Pinecone"
   - "Weaviate"
   - "Qdrant"
   - "Chroma"

5. **Clean Keywords**:
   - Remove trailing words like "etc.", "and more", "among others"
   - Keep abbreviations and acronyms (LLMs, RAG, API, etc.)
   - Keep hyphenated terms (Fine-tuning, Multi-modal)
   - Keep version numbers (Python 3.x, TensorFlow 2.x)

6. **Focus Areas**:
   - Technical skills and technologies
   - Programming languages
   - Frameworks and libraries
   - Tools and platforms
   - Methodologies and techniques
   - APIs and services

Return a JSON object with a single 'keywords' array containing ALL extracted individual keywords."""

        user_prompt = f"""Parse this job description and extract ALL individual technical keywords. Pay special attention to content in parentheses and brackets - extract both the main terms AND all individual items inside.

Job Description:
{job_description}

Examples of expected parsing:
- "Large Language Models (LLMs)" → ["Large Language Models", "LLMs"]
- "Fine-tuning (LoRA, PEFT, QLoRA, etc.)" → ["Fine-tuning", "LoRA", "PEFT", "QLoRA"]
- "Vector Databases (Pinecone, Weaviate)" → ["Vector Databases", "Pinecone", "Weaviate"]

Return JSON with 'keywords' array containing all extracted terms."""
        
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
    
    def extract_all_projects(self, resume: Dict) -> list:
        """Extract ALL projects from both 'projects' and 'experience' sections."""
        all_projects = []
        
        # Process original projects
        for proj in resume.get('projects', []):
            proj_copy = proj.copy()
            proj_copy['source'] = 'projects'
            all_projects.append(proj_copy)
        
        # Process experience descriptions and extract as projects
        for exp in resume.get('experience', []):
            exp_project = {
                'title': exp.get('title', exp.get('position', 'Professional Experience')),
                'description': exp.get('description', ''),
                'technologies': exp.get('technologies', []) if 'technologies' in exp else [],
                'company': exp.get('company', ''),
                'duration': exp.get('duration', ''),
                'source': 'experience'
            }
            all_projects.append(exp_project)
        
        return all_projects

    def score_project_relevance(self, project: Dict, job_keywords: Set[str]) -> float:
        """Score a project based on JD relevance and quality."""
        if not job_keywords:
            # Without JD, score based on content quality
            description = project.get('description', '').lower()
            title = project.get('title', '').lower()
            
            # Quality indicators
            quality_score = 0
            if len(description) > 100:  # Substantial description
                quality_score += 0.3
            if any(tech in description for tech in ['developed', 'built', 'implemented', 'designed', 'created']):
                quality_score += 0.3
            if any(impact in description for impact in ['improved', 'increased', 'reduced', 'optimized', 'enhanced']):
                quality_score += 0.2
            if any(metric in description for metric in ['%', 'users', 'performance', 'time', 'efficiency']):
                quality_score += 0.2
            
            return min(1.0, quality_score)
        
        # With JD, score based on keyword relevance
        description_lower = project.get('description', '').lower()
        title_lower = project.get('title', '').lower()
        keywords_lower = {k.lower() for k in job_keywords}
        
        # Count keyword matches in description and title
        desc_matches = sum(1 for keyword in keywords_lower if keyword in description_lower)
        title_matches = sum(1 for keyword in keywords_lower if keyword in title_lower)
        
        # Calculate base score
        total_keywords = len(keywords_lower)
        if total_keywords == 0:
            return 0.5
        
        # Weight title matches more heavily
        relevance_score = (desc_matches + (title_matches * 2)) / (total_keywords * 1.5)
        
        # Bonus for technical depth
        technical_words = ['implemented', 'developed', 'built', 'designed', 'optimized', 'automated']
        tech_bonus = sum(0.1 for word in technical_words if word in description_lower)
        
        # Bonus for impact metrics
        impact_bonus = 0.1 if any(metric in description_lower for metric in ['%', 'improved', 'increased', 'reduced']) else 0
        
        total_score = min(1.0, relevance_score + tech_bonus + impact_bonus)
        return total_score

    def select_top_2_projects(self, all_projects: list, job_keywords: Set[str]) -> list:
        """Select exactly 2 best projects based on relevance and quality."""
        if len(all_projects) == 0:
            return []
        
        # Score all projects
        scored_projects = []
        for project in all_projects:
            score = self.score_project_relevance(project, job_keywords)
            scored_projects.append((project, score))
        
        # Sort by score (highest first)
        scored_projects.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 2 projects (without scores)
        top_projects = [project for project, score in scored_projects[:2]]
        return top_projects

    def enhance_project_description(self, project: Dict, job_keywords: Set[str], has_jd: bool) -> Dict:
        """Enhance project description while staying truthful to original content."""
        
        if not has_jd:
            # Without JD, only enhance if description is too short
            original_desc = project.get('description', '')
            if len(original_desc.strip()) > 150:  # Already substantial
                return project
            
            # Enhance for page fill only
            system_prompt = """You are a professional resume writer. Enhance this project description to be more comprehensive and professional while staying completely truthful to the original content.

Rules:
1. Do NOT add fake details, metrics, or technologies not mentioned
2. Improve clarity and professional language
3. Expand on existing details without inventing new ones
4. Keep the same technical scope and achievements
5. Make it substantial enough for a professional resume
6. Use professional action verbs and clear structure
7. Do NOT add company names or specific client details

Return only the enhanced description, no title."""

            user_prompt = f"""
Original Project: {project.get('title', '')}
Original Description: {original_desc}

Enhance this description to be more professional and comprehensive while staying completely truthful to the original content. Do not invent details.
"""
        else:
            # With JD, enhance to show relevance while staying truthful
            system_prompt = """You are a professional resume writer. Enhance this project description to highlight its relevance to the job requirements while staying completely truthful to the original content.

Rules:
1. Do NOT add technologies or skills not mentioned in the original
2. Do NOT invent metrics, user numbers, or performance improvements
3. Emphasize aspects that align with job keywords if they exist in original
4. Improve professional language and clarity
5. Highlight existing technical achievements and impact
6. Use strong action verbs and clear structure
7. Stay within the scope of the original work described

Return only the enhanced description, no title."""

        user_prompt = f"""
Original Project: {project.get('title', '')}
Original Description: {project.get('description', '')}
Job Keywords: {', '.join(job_keywords)}

Enhance this description to highlight relevance to the job while staying completely truthful to the original content. Emphasize existing aspects that align with job requirements.
"""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "text"}
            )
            
            enhanced_description = response.choices[0].message.content.strip()
            
            # Create enhanced project
            enhanced_project = project.copy()
            enhanced_project["description"] = enhanced_description
            
            return enhanced_project
            
        except Exception as e:
            st.error(f"Error enhancing project description: {str(e)}")
            return project

    def generate_professional_title(self, candidate: Dict, job_keywords: Set[str] = None, job_description: str = "") -> str:
        """Generate appropriate professional title based on context."""
        
        if not job_keywords or not job_description:
            # No JD - generate based on candidate profile only
            system_prompt = """Create a professional job title based solely on the candidate's actual experience and skills.

Rules:
- Use standard industry job titles
- Match their actual experience level
- Be specific to their expertise area
- Do NOT overstate their experience
- Return ONLY the job title"""

            user_prompt = f"""
Candidate Profile:
- Current Title: {candidate.get('title', '')}
- Skills: {', '.join(candidate.get('skills', [])[:10])}
- Top Projects: {json.dumps(candidate.get('projects', [])[:2], indent=2)}

Create an appropriate professional job title based on their actual experience."""
        else:
            # With JD - align with job requirements while staying realistic
            system_prompt = """Create a professional job title that aligns with the job requirements while accurately reflecting the candidate's experience.

Rules:
- Use standard industry job titles
- Match their actual experience level
- Align with job requirements where appropriate
- Do NOT overstate their experience
- Return ONLY the job title"""

            user_prompt = f"""
Candidate Profile:
- Current Title: {candidate.get('title', '')}
- Skills: {', '.join(candidate.get('skills', [])[:10])}
- Top Projects: {json.dumps(candidate.get('projects', [])[:2], indent=2)}

Job Requirements: {', '.join(list(job_keywords)[:10])}

Create a professional job title that aligns with the job while reflecting their actual experience level."""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "text"}
            )
            
            title = response.choices[0].message.content.strip().replace('"', '').replace("'", '')
            return title
            
        except Exception as e:
            st.error(f"Error generating title: {str(e)}")
            return candidate.get('title', 'Software Professional')

    def generate_professional_summary(self, candidate: Dict, job_keywords: Set[str] = None) -> str:
        """Generate professional summary based on context."""
        
        if not job_keywords:
            # No JD - generic professional summary
            system_prompt = """Create a professional summary based solely on the candidate's profile.

Rules:
- 3-4 sentences maximum
- Use third person
- Focus on actual experience and skills
- Be specific about technologies they know
- Do NOT invent experience
- Keep professional and factual"""

            user_prompt = f"""
Candidate Profile:
- Name: {candidate.get('name', '')}
- Title: {candidate.get('title', '')}
- Skills: {', '.join(candidate.get('skills', []))}
- Projects: {json.dumps(candidate.get('projects', [])[:3], indent=2)}
- Education: {json.dumps(candidate.get('education', []), indent=2)}

Create a professional summary highlighting their actual expertise."""
        else:
            # With JD - align summary with job requirements
            system_prompt = """Create a professional summary that highlights the candidate's fit for the job while staying truthful to their actual experience.

Rules:
- 3-4 sentences maximum
- Use third person
- Emphasize relevant experience and skills
- Highlight alignment with job requirements
- Do NOT invent experience or skills
- Stay factual and professional"""

            user_prompt = f"""
Candidate Profile:
- Name: {candidate.get('name', '')}
- Title: {candidate.get('title', '')}
- Skills: {', '.join(candidate.get('skills', []))}
- Projects: {json.dumps(candidate.get('projects', [])[:3], indent=2)}
- Education: {json.dumps(candidate.get('education', []), indent=2)}

Job Requirements: {', '.join(list(job_keywords))}

Create a professional summary highlighting their relevant experience for this job."""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "text"}
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return candidate.get('summary', 'Experienced professional with diverse technical skills.')

    def optimize_skills_list(self, original_skills: list, job_keywords: Set[str]) -> list:
        """Optimize skills list by mixing original skills with top JD skills."""
        
        if not job_keywords:
            # No JD - return original skills (max 22)
            return original_skills[:22]
        
        # Convert to lowercase for comparison
        job_keywords_lower = {k.lower() for k in job_keywords}
        original_skills_lower = {s.lower(): s for s in original_skills}
        
        # Category 1: Original skills that match JD (keep original casing)
        matching_skills = []
        for keyword_lower in job_keywords_lower:
            if keyword_lower in original_skills_lower:
                matching_skills.append(original_skills_lower[keyword_lower])
        
        # Category 2: Top 5 important JD skills not in original
        top_jd_skills = []
        jd_skills_list = list(job_keywords)[:10]  # Consider top 10 JD skills
        for skill in jd_skills_list:
            if skill.lower() not in original_skills_lower and len(top_jd_skills) < 5:
                top_jd_skills.append(skill)
        
        # Category 3: Remaining original skills
        remaining_original = []
        for skill in original_skills:
            if skill.lower() not in job_keywords_lower:
                remaining_original.append(skill)
        
        # Combine: matching + top JD + remaining original
        final_skills = []
        final_skills.extend(matching_skills)  # Always include matches
        final_skills.extend(top_jd_skills)    # Add top 5 JD skills
        final_skills.extend(remaining_original)  # Add remaining original
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in final_skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique_skills.append(skill)
        
        # Limit to 22 skills
        return unique_skills[:22]

    def retailor_resume(self, original_resume: Dict, job_keywords: Set[str], job_description: str = "") -> Dict:
        """Clean, focused resume retailoring with exactly 2 projects and minimal hallucination."""
        safe_resume = convert_objectid_to_str(original_resume)
        has_jd = bool(job_description and job_description.strip())
        
        # Extract all projects from both sections
        all_projects = self.extract_all_projects(safe_resume)
        
        # Select exactly 2 best projects
        top_2_projects = self.select_top_2_projects(all_projects, job_keywords if has_jd else set())
        
        # Enhance project descriptions (truthfully)
        enhanced_projects = []
        for project in top_2_projects:
            enhanced_project = self.enhance_project_description(project, job_keywords, has_jd)
            enhanced_projects.append(enhanced_project)
        
        # Generate professional title
        safe_resume["title"] = self.generate_professional_title(
            safe_resume, 
            job_keywords if has_jd else None, 
            job_description
        )
        
        # Generate professional summary
        safe_resume["summary"] = self.generate_professional_summary(
            safe_resume, 
            job_keywords if has_jd else None
        )
        
        # Optimize skills list
        original_skills = safe_resume.get("skills", [])
        safe_resume["skills"] = self.optimize_skills_list(
            original_skills, 
            job_keywords if has_jd else set()
        )
        
        # Set exactly 2 projects
        safe_resume["projects"] = enhanced_projects
        
        # Keep certifications completely unchanged
        # (already preserved in safe_resume)
        
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