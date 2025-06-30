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
import random

class JobDescriptionAnalyzer:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=st.secrets["azure_openai"]["api_key"],
            api_version=st.secrets["azure_openai"]["api_version"],
            azure_endpoint=st.secrets["azure_openai"]["endpoint"]
        )
        
    def extract_keywords(self, job_description: str) -> Dict[str, Set[str]]:
        """Extract keywords from job description using Azure OpenAI."""
        prompt = f"""You are an AI assistant that extracts ONLY the most relevant and specific keywords from job descriptions. Focus on extracting:
1. Required technical skills and technologies
2. Programming languages  
3. Tools and frameworks
4. Key responsibilities that indicate required skills

Rules:
- Extract ONLY keywords that are explicitly mentioned in the job description
- DO NOT add any keywords that are not directly stated
- DO NOT infer or assume additional skills
- Return a JSON object with a single array field named 'keywords'
- Keep the list focused and specific
- Avoid generic terms unless explicitly required

Job Description:
{job_description}

Extract ONLY the explicitly mentioned technical keywords and return a JSON object with a 'keywords' array:"""
        
        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
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
        """Score a project based on keyword relevance, technical depth, and impact."""
        # Lowercase everything for case-insensitive comparison
        description_lower = project.get('description', '').lower()
        title_lower = project.get('title', '').lower()
        keywords_lower = {k.lower() for k in job_keywords}
        
        if not keywords_lower:
            return 0.0

        # Keyword matching score
        matches_in_desc = sum(1 for keyword in keywords_lower if keyword in description_lower)
        matches_in_title = sum(1 for keyword in keywords_lower if keyword in title_lower)
        
        # Give more weight to title matches
        keyword_score = (matches_in_desc + (matches_in_title * 2)) / len(keywords_lower)
        
        # Bonus for technical depth
        tech_words = ['api', 'database', 'backend', 'frontend', 'deploy', 'test', 'develop', 'implement', 'integrate']
        tech_bonus = 0.1 * sum(1 for word in tech_words if word in description_lower)
        
        # Bonus for impact/metrics
        impact_words = ['optimized', 'improved', 'increased', 'reduced', '%', 'latency', 'performance', 'users']
        impact_bonus = 0.1 * sum(1 for word in impact_words if word in description_lower)

        total_score = min(1.0, keyword_score + tech_bonus + impact_bonus)
        return total_score


    def generate_professional_title(self, candidate: Dict, job_keywords: Set[str], job_description: str) -> str:
        """Generate a job-specific title for the candidate based on their profile and job description."""
        prompt = f"""You are an expert HR professional and career coach specializing in job title creation. Create a specific, professional job title that accurately reflects the candidate's experience level and aligns perfectly with the provided job description.

CRITICAL RULES:
- Analyze both the candidate's experience (especially years of experience and project complexity) and the job description's requirements (e.g., "Senior Engineer," "Lead," "Junior").
- The generated title MUST be a standard, industry-recognized job title.
- The title MUST precisely match the candidate's seniority level. Do not suggest "Senior" for a junior candidate or vice-versa.
- The title MUST align with the core responsibilities and technologies mentioned in the job description.
- Your final output MUST be ONLY the job title, with no extra text, quotes, or explanations.

**Candidate Profile:**
- Current Title: {candidate.get('title', '')}
- Experience Summary: {json.dumps(candidate.get('experience', []), indent=2)}
- Project Summary: {json.dumps(candidate.get('projects')[:2], indent=2)}
- Skills: {', '.join(candidate.get('skills', []))}

**Job Description Snippet:**
{job_description[:1000]}

Generate a single, appropriate job title and return nothing else:"""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "text"}
            )
            return response.choices[0].message.content.strip().replace('"', '')
        except Exception as e:
            st.error(f"Error generating job title: {str(e)}")
            return candidate.get('title', '')

    def generate_professional_summary(self, candidate: Dict, job_keywords: Set[str]) -> str:
        """Generates a professional summary aligned with job keywords."""
        prompt = f"""You are an expert resume writer and HR professional. Create a concise, compelling, and professional summary (3-4 sentences) for a candidate, tailored to a specific job.

CRITICAL RULES:
- Write in the third person, maintaining a formal and confident tone.
- The summary must strictly be based on the candidate's profile. Do not invent or exaggerate information.
- Seamlessly weave in skills and experiences that are most relevant to the provided Job Keywords.
- The summary should highlight the candidate's key strengths and value proposition for the role.
- Return ONLY the summary paragraph. Do not include any extra text, labels, or quotation marks.

Write a professional summary for the following candidate, focusing on their fit for a job that requires these keywords: {', '.join(job_keywords)}.

**Candidate Profile:**
- Title: {candidate.get('title', '')}
- Skills: {', '.join(candidate.get('skills', []))}
- Experience: {json.dumps(candidate.get('experience', []), indent=2)}
- Projects: {json.dumps(candidate.get('projects', []), indent=2)}

Generate a 3-4 sentence professional summary based *only* on the provided information:"""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                response_format={"type": "text"}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return candidate.get('summary', '')

    def optimize_skills_list(self, original_skills: list, job_keywords: Set[str]) -> list:
        """
        Creates a balanced list of skills by reserving ~50% of slots for original skills
        and ~50% for job-relevant keywords, up to a maximum of 18.
        """
        MAX_SKILLS = 18
        # Use roughly a 50/50 split for the quota
        QUOTA_PER_CATEGORY = MAX_SKILLS // 2

        original_skills_set = {s.lower(): s for s in original_skills}
        job_keywords_set = {k.lower(): k for k in job_keywords}

        # Skills that are in both lists (highest priority)
        matching_skills = {s for s_lower, s in original_skills_set.items() if s_lower in job_keywords_set}

        # Unique original skills (not in JD)
        unique_original = [s for s_lower, s in original_skills_set.items() if s_lower not in job_keywords_set]
        random.shuffle(unique_original)

        # Unique job keywords (not in original skills)
        unique_keywords = [k for k_lower, k in job_keywords_set.items() if k_lower not in original_skills_set]
        random.shuffle(unique_keywords)

        final_skills = list(matching_skills)

        # Fill remaining slots from unique original skills, up to quota
        fill_from_original = unique_original[:max(0, QUOTA_PER_CATEGORY - len(final_skills))]
        final_skills.extend(fill_from_original)

        # Fill remaining slots from unique job keywords, up to quota
        fill_from_keywords = unique_keywords[:max(0, QUOTA_PER_CATEGORY)]
        final_skills.extend(fill_from_keywords)

        # If we are still under the max, fill with remaining skills from both pools
        if len(final_skills) < MAX_SKILLS:
            remaining_pool = [s for s in unique_original if s not in final_skills] + \
                             [k for k in unique_keywords if k not in final_skills]
            random.shuffle(remaining_pool)
            final_skills.extend(remaining_pool)

        # Ensure uniqueness and limit to max
        return list(dict.fromkeys(final_skills))[:MAX_SKILLS]

    def retailor_resume(self, original_resume: Dict, job_keywords: Set[str], job_description: str = "") -> Dict:
        """
        Retailors a resume:
        - Always enhances all project titles.
        - If JD is given: selects only relevant projects and enhances both title and description using CAR strategy with JD keywords.
        - If no JD: enhances only the project titles, leaves descriptions unchanged.
        - Ensures title and summary are present and job-specific.
        - Uses single prompt for all LLM calls.
        """
        safe_resume = convert_objectid_to_str(original_resume)
        
        # Only add JD keywords that the candidate actually demonstrates
        original_skills = list(safe_resume.get("skills", []))
        candidate_text = self._extract_candidate_text(safe_resume)
        matching_keywords = self._find_matching_keywords(job_keywords, original_skills, candidate_text)
        
        # Combine original skills with only genuinely matching JD keywords
        final_skills = list(original_skills)
        for keyword in matching_keywords:
            if keyword not in final_skills:
                final_skills.append(keyword)
        
        safe_resume["skills"] = final_skills

        all_projects = self.extract_all_projects(safe_resume)
        
        if job_description and job_keywords:
            # When JD is provided: Select only relevant projects and enhance them
            relevant_projects = self.select_relevant_projects(all_projects, job_keywords, job_description)
            enhanced_projects = []
            for proj in relevant_projects:
                # UNIVERSAL title enhancement
                enhanced_title = self.universal_enhance_project_title(proj)
                proj = proj.copy()
                proj['title'] = enhanced_title
                # Enhance description with CAR strategy using JD keywords
                enhanced_desc = self.enhance_project_description_car(proj, job_keywords, jd_given=True)
                proj['description'] = enhanced_desc
                enhanced_projects.append(proj)
        else:
            # When no JD: Enhance titles only for all projects
            enhanced_projects = []
            for proj in all_projects:
                enhanced_title = self.universal_enhance_project_title(proj)
                proj = proj.copy()
                proj['title'] = enhanced_title
                # Description remains unchanged when no JD
                enhanced_projects.append(proj)
        
        safe_resume['projects'] = enhanced_projects

        if job_description:
            safe_resume["title"] = self.generate_job_specific_title(safe_resume, job_keywords, job_description)

        if not safe_resume.get("summary"):
            if job_description:
                # Summary with job keywords when JD is provided
                summary_prompt = (
                    "Summarize the candidate's profile in 2-3 factual sentences based ONLY on the following information: skills, experience, and projects. "
                    "Use job description keywords where relevant. Do NOT invent or add any information. Return ONLY the summary, no extra text.\n"
                    f"Candidate Name: {safe_resume.get('name', '')}\n"
                    f"Skills: {', '.join(safe_resume.get('skills', []))}\n"
                    f"Projects: {json.dumps([p['title'] for p in enhanced_projects])}\n"
                    f"Experience: {json.dumps(safe_resume.get('experience', []))}\n"
                    f"Job Keywords: {', '.join(top_keywords)}"
                )
            else:
                # Generic summary when no JD is provided
                summary_prompt = (
                    "Summarize the candidate's profile in 2-3 factual sentences based ONLY on the following information: skills, experience, and projects. "
                    "Focus on their technical expertise and professional background. Do NOT invent or add any information. Return ONLY the summary, no extra text.\n"
                    f"Candidate Name: {safe_resume.get('name', '')}\n"
                    f"Skills: {', '.join(safe_resume.get('skills', []))}\n"
                    f"Projects: {json.dumps([p['title'] for p in enhanced_projects])}\n"
                    f"Experience: {json.dumps(safe_resume.get('experience', []))}"
                )
            try:
                response = self.client.chat.completions.create(
                    model=st.secrets["azure_openai"]["deployment"],
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.2,
                    response_format={"type": "text"}
                )
                safe_resume["summary"] = response.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                safe_resume["summary"] = (
                    "Experienced professional with a strong background in relevant skills and projects."
                )
        return safe_resume

    def llm_judge_project_relevance(self, project: Dict, job_keywords: Set[str], job_description: str) -> float:
        """Use LLM to judge how relevant a project is to a job description, even without direct keyword matches."""
        keywords_str = ", ".join(list(job_keywords)[:10])
        project_title = project.get('title', '')
        project_description = project.get('description', '')
        
        prompt = f"""You are an expert HR professional and technical recruiter. Your job is to evaluate how relevant a candidate's project is to a specific job requirement, even if there are no direct keyword matches.

Consider:
1. **Transferable Skills**: Does the project demonstrate skills that could transfer to the job?
2. **Technical Complexity**: Does the project show technical depth relevant to the role?
3. **Problem-Solving**: Does the project demonstrate problem-solving abilities needed for the job?
4. **Industry Relevance**: Is the project domain or technology stack somewhat related?
5. **Potential**: Could this project experience be valuable for the target role?

Rate the relevance on a scale of 0.0 to 1.0 where:
- 0.0 = Completely irrelevant, no transferable value
- 0.3 = Some transferable skills but distant relevance
- 0.5 = Moderate relevance with transferable skills
- 0.7 = High relevance with strong transferable value
- 1.0 = Perfect match, highly relevant

**Job Requirements:**
Keywords: {keywords_str}
Job Description: {job_description[:800]}

**Project to Evaluate:**
Title: {project_title}
Description: {project_description}

Return ONLY a number between 0.0 and 1.0 representing the relevance score."""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "text"}
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Extract the numeric score
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))  # Ensure it's between 0.0 and 1.0
            except ValueError:
                # If we can't parse the score, extract it from the response
                import re
                numbers = re.findall(r'\d*\.?\d+', score_text)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))
                return 0.0
                
        except Exception as e:
            st.error(f"Error in LLM judge: {str(e)}")
            return 0.0

    def select_best_closest_projects(self, all_projects: list, job_keywords: Set[str], job_description: str, max_projects: int = 2) -> list:
        """Use LLM judge to select the best/closest projects when no direct matches exist."""
        if not all_projects:
            return []
        
        # Score each project using LLM judge
        project_scores = []
        for proj in all_projects:
            score = self.llm_judge_project_relevance(proj, job_keywords, job_description)
            project_scores.append((proj, score))
        
        # Sort by relevance score (highest first) and take the top N
        project_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top projects
        return [proj for proj, score in project_scores[:max_projects]]

    def select_relevant_projects(self, all_projects: list, job_keywords: Set[str], job_description: str = "") -> list:
        """Return relevant projects (by JD keywords) or use LLM judge to select 2 best closest projects."""
        keywords_lower = {k.lower() for k in job_keywords}
        relevant = []
        
        # First, try direct keyword matching
        for proj in all_projects:
            text = (proj.get('title', '') + ' ' + proj.get('description', '')).lower()
            if any(k in text for k in keywords_lower):
                relevant.append(proj)
        
        if relevant:
            return relevant
        
        # If no direct matches and we have a job description, use LLM judge
        if job_description and job_keywords:
            st.info("No direct keyword matches found. Using AI to select the most relevant projects...")
            return self.select_best_closest_projects(all_projects, job_keywords, job_description, max_projects=2)
        
        # Fallback: pick top 2 by description length
        return sorted(all_projects, key=lambda p: len(p.get('description', '')), reverse=True)[:2]

    # Utility for strict normalization (for title comparison)
    @staticmethod
    def _normalize_title(text):
        """Normalize text for strict comparison: lowercase, remove whitespace and punctuation."""
        import string
        return ''.join(c for c in text.lower() if c not in string.whitespace + string.punctuation)

    def universal_enhance_project_title(self, project: Dict) -> str:
        """
        UNIVERSAL function that ALWAYS enhances project titles to be skill-focused and impactful.
        Works regardless of whether JD is provided or not. Guaranteed to produce a different title.
        """
        original_title = project.get('title', '').strip()
        description = project.get('description', '')
        technologies = project.get('technologies', [])
        
        # If no original title, create a basic one
        if not original_title:
            original_title = "Technical Project"
        
        prompt = f"""You are an expert at creating impactful, skill-focused project titles. Your job is to rewrite this project title to be more professional, attention-grabbing, and technology-focused.

CRITICAL REQUIREMENTS:
- The new title MUST be different from the original title
- Highlight the main technologies/skills used in the project
- Make it professional and impactful
- Use technology prefixes when applicable (e.g., "React-Based", "Python-Powered", "n8n-Driven", "AWS-Deployed")
- Focus on what makes this project technically interesting
- Return ONLY the enhanced title, no explanations

Original Title: {original_title}
Project Description: {description}
Technologies Used: {', '.join(technologies) if technologies else 'Not specified'}

Examples of good enhanced titles:
- "n8n-Based Resume Automation Pipeline" 
- "React-Powered E-commerce Platform"
- "Python-Driven Data Analytics Dashboard"
- "AWS-Deployed Microservices Architecture"
- "MongoDB-Backed Social Media Application"
- "Full-Stack Web Application with Authentication"
- "Machine Learning-Powered Recommendation System"
- "Real-Time Chat Application with WebSocket"

Create an enhanced, skill-focused title:"""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Slightly higher for more variety
                response_format={"type": "text"}
            )
            enhanced_title = response.choices[0].message.content.strip()
            
            # Remove any quotes if they exist
            enhanced_title = enhanced_title.strip('"\'')
            
            # Safety check: if somehow the same title is returned, force a change
            if self._normalize_title(enhanced_title) == self._normalize_title(original_title):
                # Extract main technology and create a forced enhancement
                main_tech = self._extract_main_technology(description, technologies)
                if main_tech:
                    enhanced_title = f"{main_tech}-Based {original_title}"
                else:
                    enhanced_title = f"Advanced {original_title}"
            
            return enhanced_title
            
        except Exception as e:
            st.error(f"Error enhancing project title: {str(e)}")
            # Robust fallback that always produces a different title
            main_tech = self._extract_main_technology(description, technologies)
            if main_tech:
                return f"{main_tech}-Based {original_title}"
            else:
                return f"Professional {original_title}"
    
    def _extract_main_technology(self, description: str, technologies: list) -> str:
        """Helper method to extract the main technology from project info."""
        # First check the technologies list
        if technologies:
            return technologies[0].title()
        
        # Then check the description for common technologies
        description_lower = description.lower()
        common_techs = [
            'n8n', 'python', 'javascript', 'react', 'node', 'nodejs', 'java', 'aws', 
            'mongodb', 'mysql', 'postgresql', 'docker', 'kubernetes', 'tensorflow', 
            'flask', 'django', 'angular', 'vue', 'spring', 'express', 'redis', 
            'elasticsearch', 'jenkins', 'git', 'llm', 'openai', 'chatgpt', 'gpt', 
            'azure', 'firebase', 'stripe', 'oauth', 'jwt', 'restapi', 'graphql', 
            'websocket', 'microservices', 'serverless', 'lambda', 'html', 'css', 
            'bootstrap', 'tailwind', 'nextjs', 'nuxtjs', 'svelte', 'php', 'laravel', 
            'symfony', 'ruby', 'rails', 'go', 'rust', 'swift', 'kotlin', 'flutter', 
            'dart', 'unity', 'unreal', 'blender'
        ]
        
        for tech in common_techs:
            if tech in description_lower:
                # Special case for n8n to keep it uppercase
                return 'n8n' if tech == 'n8n' else tech.title()
        
        return ""

    def enhance_project_description_car(self, project: Dict, job_keywords: Set[str], jd_given: bool = True) -> str:
        """Enhance project description using enhanced CAR strategy with detailed formatting requirements."""
        keywords = ", ".join(list(job_keywords)[:8])  # Increased to 8 keywords for better coverage
        original_description = project.get('description', '').strip()
        
        if not original_description:
            return original_description
            
        prompt = f"""You're a resume writing assistant. Rewrite the given project description into 6–8 **concise**, **to-the-point**, **easy-to-read sentences** using the CAR (Cause, Action, Result) strategy. Your goal is to:

- Maintain accuracy — do not hallucinate or exaggerate.
- Use provided **keywords** from the job description whenever applicable.
- Avoid fluff and background info — focus on **what was done and why it mattered**.
- Write each sentence as a **standalone impact point** suitable for resume or LinkedIn.
- **CRITICAL: Do NOT use any bullet points, symbols, dashes, arrows, or formatting markers. Write ONLY plain, direct sentences separated by line breaks.**
- **Do NOT use any markdown formatting (no **, *, _, etc.) or blank lines. Output only plain text sentences.**
- **Keep each sentence extremely concise and clear to the point.**
- If results or metrics are not given, **do not make them up**.
- Use clear, active language and relevant technical terminology.

---

**Input**
Project Description: {original_description}
Keywords: {keywords}

---

**Output**
6–8 clean, CAR-style resume points. Each should be 1–2 lines max, use keywords where appropriate, and communicate tangible work or outcomes clearly."""
        
        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "text"}
            )
            enhanced_description = response.choices[0].message.content.strip()
            
            # Ensure we return something useful even if the response is empty
            if not enhanced_description or len(enhanced_description) < 20:
                return original_description
                
            return enhanced_description
            
        except Exception as e:
            st.error(f"Error enhancing project description: {str(e)}")
            return original_description

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
        
        prompt = f"""You are an AI designed to evaluate candidate suitability for a job based on pre-extracted job description keywords. Compare the candidate's skills and projects against the job description keywords and assign a holistic match score.

Evaluation Guidelines:
1. Primary Focus (80% of score): Skills match with job requirements, Project relevance and implementation of required technologies
2. Secondary Focus (20% of score): Education relevance, Experience relevance, Certifications

Scoring Rules:
- Score range: 1-100
- Focus on exact matches and closely related technologies
- Higher scores for candidates with multiple, highly relevant matches
- Lower scores for partial matches or minimal alignment
- Do not assume or hallucinate missing information
- Explicitly mention missing required skills in the reason

Status Rules: "Accepted" if score > 70, "Rejected" if score ≤ 70

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
                messages=[{"role": "user", "content": prompt}],
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
    
    def enhance_project_description_car(self, project: Dict, job_keywords: Set[str], jd_given: bool = True) -> str:
        """Enhance project description using enhanced CAR strategy with detailed formatting requirements."""
        keywords = ", ".join(list(job_keywords)[:8])  # Increased to 8 keywords for better coverage
        original_description = project.get('description', '').strip()
        
        if not original_description:
            return original_description
            
        prompt = f"""You're a resume writing assistant. Rewrite the given project description into 6–8 **concise**, **to-the-point**, **easy-to-read sentences** using the CAR (Cause, Action, Result) strategy. Your goal is to:

- Maintain accuracy — do not hallucinate or exaggerate.
- Use provided **keywords** from the job description whenever applicable.
- Avoid fluff and background info — focus on **what was done and why it mattered**.
- Write each sentence as a **standalone impact point** suitable for resume or LinkedIn.
- **CRITICAL: Do NOT use any bullet points, symbols, dashes, arrows, or formatting markers. Write ONLY plain, direct sentences separated by line breaks.**
- **Do NOT use any markdown formatting (no **, *, _, etc.) or blank lines. Output only plain text sentences.**
- **Keep each sentence extremely concise and clear to the point.**
- If results or metrics are not given, **do not make them up**.
- Use clear, active language and relevant technical terminology.

---

**Input**
Project Description: {original_description}
Keywords: {keywords}

---

**Output**
6–8 clean, CAR-style resume points. Each should be 1–2 lines max, use keywords where appropriate, and communicate tangible work or outcomes clearly."""
        
        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "text"}
            )
            enhanced_description = response.choices[0].message.content.strip()
            
            # Ensure we return something useful even if the response is empty
            if not enhanced_description or len(enhanced_description) < 20:
                return original_description
                
            return enhanced_description
            
        except Exception as e:
            st.error(f"Error enhancing project description: {str(e)}")
            return original_description
    
    def generate_job_specific_title(self, candidate: Dict, job_keywords: Set[str], job_description: str) -> str:
        """Generate a job-specific title for the candidate based on their profile and job description."""
        prompt = f"""You are an expert HR professional specializing in job title creation. Create a specific, professional job title that accurately reflects the candidate's experience and aligns with the job description requirements.

Rules:
- Use industry-standard job titles
- Match the candidate's actual experience level (don't overstate)
- Be specific to the role and industry
- Use ONLY information from the candidate's profile and job description
- Return ONLY the job title, no explanation

Candidate Profile:
- Name: {candidate.get('name', '')}
- Current Title: {candidate.get('title', '')}
- Skills: {', '.join(candidate.get('skills', []))}
- Projects: {json.dumps(candidate.get('projects', []), indent=2)}
- Education: {json.dumps(candidate.get('education', []), indent=2)}

Job Description Keywords: {', '.join(job_keywords)}
Job Description: {job_description}

Generate a professional job title that matches their experience level and aligns with the job requirements:"""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "text"}
            )
            
            title = response.choices[0].message.content.strip()
            return title
            
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

    def universal_enhance_project_title(self, project: Dict) -> str:
        """
        UNIVERSAL function that ALWAYS enhances project titles to be skill-focused and impactful.
        Works regardless of whether JD is provided or not. Guaranteed to produce a different title.
        """
        original_title = project.get('title', '').strip()
        description = project.get('description', '')
        technologies = project.get('technologies', [])
        
        # If no original title, create a basic one
        if not original_title:
            original_title = "Technical Project"
        
        prompt = f"""You are an expert at creating impactful, skill-focused project titles. Your job is to rewrite this project title to be more professional, attention-grabbing, and technology-focused.

CRITICAL REQUIREMENTS:
- The new title MUST be different from the original title
- Highlight the main technologies/skills used in the project
- Make it professional and impactful
- Use technology prefixes when applicable (e.g., "React-Based", "Python-Powered", "n8n-Driven", "AWS-Deployed")
- Focus on what makes this project technically interesting
- Return ONLY the enhanced title, no explanations

Original Title: {original_title}
Project Description: {description}
Technologies Used: {', '.join(technologies) if technologies else 'Not specified'}

Examples of good enhanced titles:
- "n8n-Based Resume Automation Pipeline" 
- "React-Powered E-commerce Platform"
- "Python-Driven Data Analytics Dashboard"
- "AWS-Deployed Microservices Architecture"
- "MongoDB-Backed Social Media Application"
- "Full-Stack Web Application with Authentication"
- "Machine Learning-Powered Recommendation System"
- "Real-Time Chat Application with WebSocket"

Create an enhanced, skill-focused title:"""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Slightly higher for more variety
                response_format={"type": "text"}
            )
            enhanced_title = response.choices[0].message.content.strip()
            
            # Remove any quotes if they exist
            enhanced_title = enhanced_title.strip('"\'')
            
            # Safety check: if somehow the same title is returned, force a change
            if self._normalize_title(enhanced_title) == self._normalize_title(original_title):
                # Extract main technology and create a forced enhancement
                main_tech = self._extract_main_technology(description, technologies)
                if main_tech:
                    enhanced_title = f"{main_tech}-Based {original_title}"
                else:
                    enhanced_title = f"Advanced {original_title}"
            
            return enhanced_title
            
        except Exception as e:
            st.error(f"Error enhancing project title: {str(e)}")
            # Robust fallback that always produces a different title
            main_tech = self._extract_main_technology(description, technologies)
            if main_tech:
                return f"{main_tech}-Based {original_title}"
            else:
                return f"Professional {original_title}"
    
    def _extract_main_technology(self, description: str, technologies: list) -> str:
        """Helper method to extract the main technology from project info."""
        # First check the technologies list
        if technologies:
            return technologies[0].title()
        
        # Then check the description for common technologies
        description_lower = description.lower()
        common_techs = [
            'n8n', 'python', 'javascript', 'react', 'node', 'nodejs', 'java', 'aws', 
            'mongodb', 'mysql', 'postgresql', 'docker', 'kubernetes', 'tensorflow', 
            'flask', 'django', 'angular', 'vue', 'spring', 'express', 'redis', 
            'elasticsearch', 'jenkins', 'git', 'llm', 'openai', 'chatgpt', 'gpt', 
            'azure', 'firebase', 'stripe', 'oauth', 'jwt', 'restapi', 'graphql', 
            'websocket', 'microservices', 'serverless', 'lambda', 'html', 'css', 
            'bootstrap', 'tailwind', 'nextjs', 'nuxtjs', 'svelte', 'php', 'laravel', 
            'symfony', 'ruby', 'rails', 'go', 'rust', 'swift', 'kotlin', 'flutter', 
            'dart', 'unity', 'unreal', 'blender'
        ]
        
        for tech in common_techs:
            if tech in description_lower:
                # Special case for n8n to keep it uppercase
                return 'n8n' if tech == 'n8n' else tech.title()
        
        return ""

    def _extract_candidate_text(self, resume: Dict) -> str:
        """Extract all text content from candidate's resume for skill matching."""
        text_parts = []
        
        # Add project descriptions and titles
        for proj in resume.get('projects', []):
            text_parts.append(proj.get('title', ''))
            text_parts.append(proj.get('description', ''))
            text_parts.extend(proj.get('technologies', []))
        
        # Add experience descriptions and titles
        for exp in resume.get('experience', []):
            text_parts.append(exp.get('title', ''))
            text_parts.append(exp.get('position', ''))
            text_parts.append(exp.get('description', ''))
            text_parts.extend(exp.get('technologies', []))
        
        # Add education information
        for edu in resume.get('education', []):
            text_parts.append(edu.get('degree', ''))
            text_parts.append(edu.get('institution', ''))
            text_parts.append(edu.get('field', ''))
        
        # Add certifications
        for cert in resume.get('certifications', []):
            if isinstance(cert, dict):
                text_parts.append(cert.get('title', ''))
                text_parts.append(cert.get('issuer', ''))
            else:
                text_parts.append(str(cert))
        
        # Add summary
        text_parts.append(resume.get('summary', ''))
        
        return ' '.join(text_parts).lower()

    def _find_matching_keywords(self, job_keywords: Set[str], original_skills: list, candidate_text: str) -> list:
        """Find JD keywords that the candidate actually demonstrates in their background."""
        matching_keywords = []
        original_skills_lower = {skill.lower() for skill in original_skills}
        
        for keyword in job_keywords:
            keyword_lower = keyword.lower()
            
            # Skip if already in original skills
            if keyword_lower in original_skills_lower:
                continue
            
            # Check if keyword appears in candidate's projects, experience, etc.
            if keyword_lower in candidate_text:
                matching_keywords.append(keyword)
        
        # Limit to top 3-5 most relevant matching keywords to avoid skill inflation
        return matching_keywords[:5]

    def llm_judge_project_relevance(self, project: Dict, job_keywords: Set[str], job_description: str) -> float:
        """Use LLM to judge how relevant a project is to a job description, even without direct keyword matches."""
        keywords_str = ", ".join(list(job_keywords)[:10])
        project_title = project.get('title', '')
        project_description = project.get('description', '')
        
        prompt = f"""You are an expert HR professional and technical recruiter. Your job is to evaluate how relevant a candidate's project is to a specific job requirement, even if there are no direct keyword matches.

Consider:
1. **Transferable Skills**: Does the project demonstrate skills that could transfer to the job?
2. **Technical Complexity**: Does the project show technical depth relevant to the role?
3. **Problem-Solving**: Does the project demonstrate problem-solving abilities needed for the job?
4. **Industry Relevance**: Is the project domain or technology stack somewhat related?
5. **Potential**: Could this project experience be valuable for the target role?

Rate the relevance on a scale of 0.0 to 1.0 where:
- 0.0 = Completely irrelevant, no transferable value
- 0.3 = Some transferable skills but distant relevance
- 0.5 = Moderate relevance with transferable skills
- 0.7 = High relevance with strong transferable value
- 1.0 = Perfect match, highly relevant

**Job Requirements:**
Keywords: {keywords_str}
Job Description: {job_description[:800]}

**Project to Evaluate:**
Title: {project_title}
Description: {project_description}

Return ONLY a number between 0.0 and 1.0 representing the relevance score."""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "text"}
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Extract the numeric score
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))  # Ensure it's between 0.0 and 1.0
            except ValueError:
                # If we can't parse the score, extract it from the response
                import re
                numbers = re.findall(r'\d*\.?\d+', score_text)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))
                return 0.0
                
        except Exception as e:
            st.error(f"Error in LLM judge: {str(e)}")
            return 0.0

    def select_best_closest_projects(self, all_projects: list, job_keywords: Set[str], job_description: str, max_projects: int = 2) -> list:
        """Use LLM judge to select the best/closest projects when no direct matches exist."""
        if not all_projects:
            return []
        
        # Score each project using LLM judge
        project_scores = []
        for proj in all_projects:
            score = self.llm_judge_project_relevance(proj, job_keywords, job_description)
            project_scores.append((proj, score))
        
        # Sort by relevance score (highest first) and take the top N
        project_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top projects
        return [proj for proj, score in project_scores[:max_projects]]

    def select_relevant_projects(self, all_projects: list, job_keywords: Set[str], job_description: str = "") -> list:
        """Return relevant projects (by JD keywords) or use LLM judge to select 2 best closest projects."""
        keywords_lower = {k.lower() for k in job_keywords}
        relevant = []
        
        # First, try direct keyword matching
        for proj in all_projects:
            text = (proj.get('title', '') + ' ' + proj.get('description', '')).lower()
            if any(k in text for k in keywords_lower):
                relevant.append(proj)
        
        if relevant:
            return relevant
        
        # If no direct matches and we have a job description, use LLM judge
        if job_description and job_keywords:
            st.info("No direct keyword matches found. Using AI to select the most relevant projects...")
            return self.select_best_closest_projects(all_projects, job_keywords, job_description, max_projects=2)
        
        # Fallback: pick top 2 by description length
        return sorted(all_projects, key=lambda p: len(p.get('description', '')), reverse=True)[:2]

    @staticmethod
    def _normalize_title(text):
        """Normalize text for strict comparison: lowercase, remove whitespace and punctuation."""
        import string
        return ''.join(c for c in text.lower() if c not in string.whitespace + string.punctuation)

    def retailor_resume(self, original_resume: Dict, job_keywords: Set[str], job_description: str = "") -> Dict:
        """
        Retailor the resume based on job requirements:
        - Always enhances all project titles.
        - If JD is given: selects only relevant projects and enhances both title and description using CAR strategy with JD keywords.
        - If no JD: enhances only the project titles, leaves descriptions unchanged.
        - Keeps all original skills unchanged and maintains project relevance focus.
        """
        safe_resume = convert_objectid_to_str(original_resume)
        
        # Generate job-specific title
        if job_description:
            safe_resume["title"] = self.generate_job_specific_title(safe_resume, job_keywords, job_description)
        
        # Extract ALL projects from both projects and experience sections
        all_projects = self.extract_all_projects(safe_resume)
        
        if job_description and job_keywords:
            # When JD is provided: Select only relevant projects and enhance them
            relevant_projects = self.select_relevant_projects(all_projects, job_keywords, job_description)
            enhanced_projects = []
            for proj in relevant_projects:
                # UNIVERSAL title enhancement
                enhanced_title = self.universal_enhance_project_title(proj)
                proj_copy = proj.copy()
                proj_copy['title'] = enhanced_title
                # Enhance description with CAR strategy using JD keywords
                enhanced_desc = self.enhance_project_description_car(proj, job_keywords, jd_given=True)
                proj_copy['description'] = enhanced_desc
                enhanced_projects.append(proj_copy)
        else:
            # When no JD: Enhance titles only for all projects
            enhanced_projects = []
            for proj in all_projects:
                enhanced_title = self.universal_enhance_project_title(proj)
                proj_copy = proj.copy()
                proj_copy['title'] = enhanced_title
                # Description remains unchanged when no JD
                enhanced_projects.append(proj_copy)
        
        # Update the resume with enhanced project titles (and descriptions if JD provided)
        safe_resume['projects'] = enhanced_projects
        
        # --- Retailor the resume (keeping all skills, only modifying projects) ---
        prompt = f"""You are an AI assistant that retailors resumes to match a job description. Your task is to:

1. KEEP ORIGINAL SKILLS AS PROVIDED - Copy the entire 'skills' list exactly as provided in the original resume
2. KEEP PROJECT TITLES AS PROVIDED - The project titles have already been enhanced, keep them exactly as provided
3. EXTRACT ALL PROJECTS from BOTH 'projects' and 'experience' sections - Convert work experience descriptions into project format and include them along with existing projects
4. Rewrite the 'summary' field to be a 2-4 sentence summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords
5. Add or update a 'title' field, inferring a proper professional job title based on the job description and their experience/skills
6. Do NOT add, invent, or hallucinate any new skills, projects, or summary content - only extract and restructure existing information
7. Return the complete resume in the exact same JSON format

Job Keywords: {json.dumps(list(job_keywords))}

Original Resume: {json.dumps(safe_resume, indent=2)}

Return the complete resume with original skills kept as provided, ALL projects (from both sources), updated summary, and appropriate title:"""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={ "type": "json_object" }
            )
            # Parse the response
            retailored_resume = json.loads(response.choices[0].message.content.strip())
            
            # IMPORTANT: Only add JD keywords that the candidate actually demonstrates
            original_skills = list(safe_resume.get("skills", []))
            candidate_text = self._extract_candidate_text(safe_resume)
            matching_keywords = self._find_matching_keywords(job_keywords, original_skills, candidate_text)
            
            # Priority 1: Original skills that match job keywords (highest priority)
            job_keywords_lower = {k.lower() for k in job_keywords}
            original_skills_lower = {s.lower(): s for s in original_skills}
            matching_skills = []
            for keyword_lower in job_keywords_lower:
                if keyword_lower in original_skills_lower:
                    matching_skills.append(original_skills_lower[keyword_lower])
            
            # Priority 2: Only genuinely demonstrated JD keywords (not all JD keywords)
            demonstrated_keywords = matching_keywords
            
            # Priority 3: Remaining original skills (non-matching ones)
            remaining_skills = []
            for skill in original_skills:
                if skill.lower() not in job_keywords_lower:
                    remaining_skills.append(skill)
            
            # Combine skills with priority order and limit to 18
            prioritized_skills = matching_skills + demonstrated_keywords + remaining_skills
            
            # Limit to maximum 18 skills
            final_skills = prioritized_skills[:18]
            
            retailored_resume["skills"] = final_skills
            
            # Ensure enhanced project titles are preserved
            retailored_resume["projects"] = enhanced_projects
            
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