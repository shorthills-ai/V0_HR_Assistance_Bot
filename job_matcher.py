import json
import re
import random
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
    
    def expand_project_description(self, project: Dict, job_keywords: Set[str]) -> str:
        """Expand a project description based on its title and minimal description, aligning with JD."""
        system_prompt = """
You are a professional technical resume writer. Your task is to convert raw project data into a concise, impactful, and metric-driven project description suitable for a resume, specifically tailored to match the job description.

Follow these rules:

1. Output 4 to 7 achievement sentences separated by periods. Each sentence should be 15-30 words maximum.
2. The first line compulsorily has to be a professionally retailored title of the project, do this for every project and make sure it has only first letter of each word capitalised.
    -> If its a work experience, make sure not to include company name in the title and give a generic title to the specific project.
    -> DO NOT repeat this title anywhere in the description that follows.
3. Each achievement sentence must start with a strong past-tense action verb (Engineered, Built, Automated, Designed, Achieved, etc.).
4. Each achievement sentence should be a complete, standalone sentence that:
 -> summarizes one specific accomplishment with technologies used and impact achieved
 -> includes specific results using clear metrics — e.g., "93% coverage," "reduced latency by 40%," "automated 80% of manual work"
 -> explains the business or user impact in measurable terms
 -> mentions relevant technologies and methodologies from the job keywords
5. CRITICAL: End each sentence with a period (.) - this is essential for proper formatting.
6. Write as a continuous paragraph with sentences separated by periods and spaces.
7. Do NOT use bullet symbols, dashes, or line breaks - write as flowing sentences with periods.
8. IMPORTANT: Align the project description with the job keywords provided. Emphasize technologies, skills, and experiences that match the job requirements.
9. If the project uses any of the job keywords, make sure to explicitly mention them and show how they were applied effectively.

Return ONLY the professional project title on the first line, followed by a paragraph of 4-7 sentences separated by periods (without repeating the title).

Example format:
Enhanced E-Commerce Platform
Engineered responsive web application using React, Node.js, and MongoDB, serving 10,000+ daily users. Implemented payment gateway integration with Stripe API, increasing conversion rates by 25%. Optimized database queries and API responses, reducing page load time by 40%. Built automated testing suite with Jest and Cypress, achieving 95% code coverage.
"""

        user_prompt = f"""
Project Title: {project['title']}
Raw Description: {project['description']}
Technologies: {', '.join(project.get('technologies', []))}
Job Keywords to Align With: {', '.join(job_keywords)}

Using the system instructions, create a professional project title on the first line, then a paragraph of 4-7 sentences (separated by periods) that summarize the project and align it with the job requirements.

Required format:
[Professional Project Title]
[Action verb] [description with technologies and impact]. [Action verb] [description with metrics/results]. [Action verb] [description highlighting job keyword alignment]. [Additional sentences as needed].
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

    def extract_all_projects(self, resume: Dict) -> list:
        """Extract ALL projects from both 'projects' and 'experience' sections."""
        all_projects = []
        
        # Process original projects
        for proj in resume.get('projects', []):
            all_projects.append(proj)
        
        # Process experience descriptions and extract as projects
        for exp in resume.get('experience', []):
            # Create a project-like dict from experience
            exp_project = {
                'title': exp.get('title', exp.get('position', 'Professional Experience')),
                'description': exp.get('description', ''),
                'technologies': exp.get('technologies', []) if 'technologies' in exp else [],
                'company': exp.get('company', ''),
                'duration': exp.get('duration', ''),
                'source': 'experience'  # Mark source for reference
            }
            all_projects.append(exp_project)
        
        return all_projects

    def score_project_relevance(self, enhanced_description: str, job_keywords: Set[str]) -> float:
        """Score a project based on how well its enhanced description aligns with job keywords."""
        description_lower = enhanced_description.lower()
        keywords_lower = {k.lower() for k in job_keywords}
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords_lower if keyword in description_lower)
        
        # Calculate score (0-1 scale)
        if len(keywords_lower) == 0:
            return 0.5  # Neutral score if no keywords
        
        # Base score from keyword matches
        keyword_score = matches / len(keywords_lower)
        
        # Bonus for multiple occurrences and context
        bonus = 0
        for keyword in keywords_lower:
            occurrences = description_lower.count(keyword)
            if occurrences > 1:
                bonus += 0.1 * (occurrences - 1)
        
        # Bonus for technical depth indicators
        technical_indicators = ['implemented', 'engineered', 'optimized', 'automated', 'designed', 'built', 'developed']
        technical_score = sum(0.05 for indicator in technical_indicators if indicator in description_lower)
        
        # Bonus for metrics/impact indicators
        impact_indicators = ['%', 'increased', 'reduced', 'improved', 'achieved', 'users', 'performance']
        impact_score = sum(0.05 for indicator in impact_indicators if indicator in description_lower)
        
        total_score = min(1.0, keyword_score + bonus + technical_score + impact_score)
        return total_score

    def retailor_resume(self, original_resume: Dict, job_keywords: Set[str], job_description: str = "") -> Dict:
        """Retailor the resume to keep all original skills unchanged and only modify projects based on job relevance."""
        safe_resume = convert_objectid_to_str(original_resume)
        
        # Generate job-specific title
        if job_description:
            safe_resume["title"] = self.generate_job_specific_title(safe_resume, job_keywords, job_description)
        
        # Extract ALL projects from both projects and experience sections
        all_projects = self.extract_all_projects(safe_resume)
        
        # --- Retailor the resume (keeping all skills, only modifying projects) ---
        system_prompt = """
You are an AI assistant that retailors resumes to match a job description.
Your ONLY task is to:
- KEEP ORIGINAL SKILLS BUT PRIORITIZE THEM - The skills list will be post-processed to prioritize job-relevant skills and limit to 22 maximum. You should keep the original skills as provided.
- INCLUDE ALL PROJECTS - Both job-relevant and non-relevant projects from 'projects' and 'experience' sections will be included and enhanced post-processing. You should keep the original projects as provided.
- KEEP CERTIFICATIONS UNCHANGED - Copy the entire 'certifications' section exactly as provided in the original resume. Do NOT modify, add, remove, or change any certification details.
- Rewrite the 'summary' field to be a concise, job-specific summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords
- Add or update a 'title' field in the resume JSON, inferring a proper, professional job title for the candidate based on the job description and their experience/skills. The title should be a realistic job title (e.g., 'Frontend Developer', 'Data Scientist', 'Project Manager'), not just a single keyword or technology. Do not leave the title blank. If unsure, use the most relevant job title from the job description.
- Extract projects from BOTH 'projects' and 'experience' sections - work experience descriptions containing projects or achievements should be converted to project format and included in the projects section.
- Do NOT add, invent, or hallucinate any new skills, projects, or summary content.
- Do NOT change or add any other fields except 'title', 'projects', and 'summary'. KEEP 'skills' and 'certifications' exactly as provided in the original resume (post-processing will handle skills prioritization and limiting to 22 skills).
- The output must be the same JSON structure as the input, but with projects (from both sources), summary, and title updated as above.
- Include ALL projects - post-processing will handle prioritization and enhancement based on job relevance.
- Do not change the order or content of any other fields.
"""
        user_prompt = f"""
Job Keywords:
{json.dumps(list(job_keywords))}

Original Resume:
{json.dumps(safe_resume, indent=2)}

Instructions:
1. KEEP ORIGINAL SKILLS AS PROVIDED - Copy the entire 'skills' list exactly as provided in the original resume (post-processing will prioritize and limit to 22 most relevant skills).
2. KEEP CERTIFICATIONS UNCHANGED - Copy the entire 'certifications' section exactly as provided in the original resume without any modifications.
3. EXTRACT ALL PROJECTS from BOTH 'projects' and 'experience' sections - Convert work experience descriptions into project format and include them along with existing projects. Include ALL projects regardless of job keyword relevance (post-processing will handle prioritization and enhancement).
4. Rewrite the 'summary' field to be a 2-4 sentence summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords.
5. Do NOT add or invent any new skills, projects, certifications, or summary content - only extract and restructure existing information.
6. Return the complete resume in the exact same JSON format, with the original skills and certifications kept as provided, ALL projects (from both sources), and new summary.
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
            
            # ENHANCED: Balanced mix of original skills and job keywords (max 22)
            original_skills = list(safe_resume.get("skills", []))
            job_keywords_list = list(job_keywords)
            
            # Convert to lowercase for case-insensitive comparison
            job_keywords_lower = {k.lower() for k in job_keywords_list}
            original_skills_lower = {s.lower(): s for s in original_skills}  # Keep original casing
            
            # Category 1: Original skills that match job keywords (MUST include)
            matching_skills = []
            for keyword_lower in job_keywords_lower:
                if keyword_lower in original_skills_lower:
                    matching_skills.append(original_skills_lower[keyword_lower])
            
            # Category 2: Important missing job keywords (selective)
            missing_keywords = []
            for keyword in job_keywords_list:
                if keyword.lower() not in original_skills_lower:
                    missing_keywords.append(keyword)
            
            # Category 3: Remaining original skills (preserve candidate's expertise)
            remaining_original_skills = []
            for skill in original_skills:
                if skill.lower() not in job_keywords_lower:
                    remaining_original_skills.append(skill)
            
            # ENHANCED: Balanced mix with complete randomization for better distribution
            max_skills = 22
            
            # Step 1: Combine all skills into categories
            all_skills_pool = []
            
            # Add matching skills (highest priority - always include)
            all_skills_pool.extend([(skill, 'matching') for skill in matching_skills])
            
            # Add remaining original skills 
            all_skills_pool.extend([(skill, 'original') for skill in remaining_original_skills])
            
            # Add missing job keywords (selective - take about half)
            selected_missing = missing_keywords[:len(missing_keywords)//2 + 2]  # Take roughly half + 2 extra
            all_skills_pool.extend([(skill, 'job_keyword') for skill in selected_missing])
            
            # Step 2: Shuffle the entire pool randomly for complete mixing
            random.shuffle(all_skills_pool)
            
            # Step 3: Select up to 22 skills while avoiding duplicates
            seen = set()
            final_skills = []
            
            for skill, source in all_skills_pool:
                if skill.lower() not in seen and len(final_skills) < max_skills:
                    seen.add(skill.lower())
                    final_skills.append(skill)
            
            # Step 4: If we still need more skills, add any remaining ones
            if len(final_skills) < max_skills:
                remaining_space = max_skills - len(final_skills)
                leftover_skills = [k for k in missing_keywords[len(missing_keywords)//2 + 2:] 
                                 if k.lower() not in seen]
                random.shuffle(leftover_skills)
                
                for skill in leftover_skills[:remaining_space]:
                    if skill.lower() not in seen:
                        seen.add(skill.lower())
                        final_skills.append(skill)
            
            retailored_resume["skills"] = final_skills[:22]
            
            # --- New Logic: Rewrite ALL projects based on JD, then select the BEST ones ---
            
            # Step 1: Rewrite ALL projects based on job description
            enhanced_projects_with_scores = []
            for i, project in enumerate(all_projects):
                # Rewrite project description based on JD
                enhanced_description = self.expand_project_description(project, job_keywords)
                
                # Extract title and description from AI response
                lines = enhanced_description.split('\n', 1)  # Split into title and rest
                if len(lines) >= 2:
                    ai_generated_title = lines[0].strip()  # First line is the AI-generated title
                    actual_description = lines[1].strip()  # Rest is the description
                else:
                    # Fallback if only one line
                    ai_generated_title = lines[0].strip() if lines else project.get('title', 'Project')
                    actual_description = ""
                
                # Score the enhanced description
                relevance_score = self.score_project_relevance(actual_description, job_keywords)
                
                # Create enhanced project with AI-generated title and cleaned description
                enhanced_project = project.copy()
                enhanced_project["title"] = ai_generated_title  # Use AI-generated title
                enhanced_project["description"] = actual_description  # Use description without title
                enhanced_project["_relevance_score"] = relevance_score
                
                enhanced_projects_with_scores.append(enhanced_project)
                time.sleep(1)  # Rate limiting for API calls
            
            # Step 2: Sort by relevance score (best first)
            enhanced_projects_with_scores.sort(key=lambda x: x["_relevance_score"], reverse=True)
            
            # Step 3: Select top 3 projects for final resume (maximum limit)
            max_projects = min(3, len(enhanced_projects_with_scores))
            final_projects = enhanced_projects_with_scores[:max_projects]
            
            # Step 4: Remove the score field before final output
            for project in final_projects:
                project.pop("_relevance_score", None)
            
            retailored_resume["projects"] = final_projects
     
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