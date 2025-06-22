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

    def enhance_project_description(self, project: Dict, job_keywords: Set[str]) -> Dict:
        """Enhance a project description to be professional and quantified, aligning with job keywords."""
        system_prompt = """You are an expert technical resume writer. Your task is to enhance a project description to be professional, quantified, and aligned with job requirements, based ONLY on the information provided.

CRITICAL RULES:
1.  **DO NOT HALLUCINATE**: Do not invent new features, technologies, or metrics not present or clearly implied in the original text. You can, however, rephrase to highlight impact.
2.  **QUANTIFY ACHIEVEMENTS**: If the original text mentions improvements (e.g., "made it faster"), translate this into a plausible, specific metric (e.g., "Reduced server response time by 30%"). If no improvement is mentioned, focus on the scale or complexity (e.g., "Engineered a system to handle over 10,000 concurrent users").
3.  **USE STRONG ACTION VERBS**: Start each bullet point with a powerful action verb (e.g., Engineered, Architected, Implemented, Optimized, Deployed, Automated).
4.  **ALIGN WITH KEYWORDS**: Seamlessly integrate relevant `Job Keywords` into the description where they genuinely fit the context of the project.
5.  **BE TECHNICALLY SPECIFIC**: Mention specific technologies from the `Original Technologies` list where relevant within the bullet points to add technical depth.
6.  **FORMAT**: Return a single string. The first line MUST be a professional, retailored title for the project. Subsequent lines MUST be bullet points, each starting with '•'. Aim for 3-5 impactful bullet points.

Example:
Original: {title: "Shop App", desc: "Made an app for shopping. It has a product screen and add to cart. Used flutter and firebase.", technologies: ["Flutter", "Firebase", "State Management"]}
Keywords: ["Mobile Development", "Firebase", "User Authentication"]
Result:
Cross-Platform E-Commerce Mobile Application
• Engineered a feature-rich shopping application using Flutter and Firebase, delivering a seamless user experience for both iOS and Android platforms.
• Implemented secure user authentication and real-time cart management leveraging Firebase, ensuring data integrity and a smooth checkout process.
• Developed a responsive and intuitive user interface with advanced state management techniques, resulting in a highly performant and user-friendly application.
• Integrated core e-commerce functionalities, including product catalogs, detailed product views, and a streamlined "Add to Cart" workflow.
"""

        user_prompt = f"""Enhance the following project based on the system rules.

**Original Project Title**: {project.get('title', '')}
**Original Description**: {project.get('description', '')}
**Original Technologies**: {', '.join(project.get('technologies', []))}

**Job Keywords to Align With**: {', '.join(job_keywords)}

Return the enhanced project as a single string, with the title on the first line, followed by 3-5 bullet points.
"""

        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                response_format={"type": "text"}
            )
            
            enhanced_text = response.choices[0].message.content.strip()
            
            # Split title and description
            lines = enhanced_text.split('\n')
            enhanced_title = lines[0].strip()
            enhanced_description = "\n".join(lines[1:]).strip()

            enhanced_project = project.copy()
            enhanced_project['title'] = enhanced_title
            enhanced_project['description'] = enhanced_description
            
            return enhanced_project
            
        except Exception as e:
            st.error(f"Error enhancing project description: {str(e)}")
            return project # Return original project on error
            
    def generate_professional_title(self, candidate: Dict, job_keywords: Set[str], job_description: str) -> str:
        """Generate a job-specific title for the candidate based on their profile and job description."""
        system_prompt = """You are an expert HR professional and career coach specializing in job title creation. Your task is to create a specific, professional job title that accurately reflects the candidate's experience level and aligns perfectly with the provided job description.

CRITICAL RULES:
- Analyze both the candidate's experience (especially years of experience and project complexity) and the job description's requirements (e.g., "Senior Engineer," "Lead," "Junior").
- The generated title MUST be a standard, industry-recognized job title.
- The title MUST precisely match the candidate's seniority level. Do not suggest "Senior" for a junior candidate or vice-versa.
- The title MUST align with the core responsibilities and technologies mentioned in the job description.
- Your final output MUST be ONLY the job title, with no extra text, quotes, or explanations.
"""

        user_prompt = f"""Analyze the candidate's profile and the job description to generate the most appropriate and specific job title.

**Candidate Profile:**
- Current Title: {candidate.get('title', '')}
- Experience Summary: {json.dumps(candidate.get('experience', []), indent=2)}
- Project Summary: {json.dumps(candidate.get('projects')[:2], indent=2)}
- Skills: {', '.join(candidate.get('skills', []))}

**Job Description Snippet:**
{job_description[:1000]}

Generate a single, appropriate job title and return nothing else.
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
            return response.choices[0].message.content.strip().replace('"', '')
        except Exception as e:
            st.error(f"Error generating job title: {str(e)}")
            return candidate.get('title', '')

    def generate_professional_summary(self, candidate: Dict, job_keywords: Set[str]) -> str:
        """Generates a professional summary aligned with job keywords."""
        system_prompt = """You are an expert resume writer and HR professional. Your task is to create a concise, compelling, and professional summary (3-4 sentences) for a candidate, tailored to a specific job.

CRITICAL RULES:
- Write in the third person, maintaining a formal and confident tone.
- The summary must strictly be based on the candidate's profile. Do not invent or exaggerate information.
- Seamlessly weave in skills and experiences that are most relevant to the provided `Job Keywords`.
- The summary should highlight the candidate's key strengths and value proposition for the role.
- Return ONLY the summary paragraph. Do not include any extra text, labels, or quotation marks.
"""
        user_prompt = f"""Write a professional summary for the following candidate, focusing on their fit for a job that requires these keywords: {', '.join(job_keywords)}.

**Candidate Profile:**
- Title: {candidate.get('title', '')}
- Skills: {', '.join(candidate.get('skills', []))}
- Experience: {json.dumps(candidate.get('experience', []), indent=2)}
- Projects: {json.dumps(candidate.get('projects', []), indent=2)}

Generate a 3-4 sentence professional summary based *only* on the provided information.
"""
        try:
            response = self.client.chat.completions.create(
                model=st.secrets["azure_openai"]["deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
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
        and ~50% for job-relevant keywords, up to a maximum of 22.
        """
        MAX_SKILLS = 22
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
        Retailors a resume by extracting all projects, scoring them, selecting the best ones,
        and enhancing their descriptions, while also updating title, summary, and skills.
        """
        safe_resume = convert_objectid_to_str(original_resume)
        has_jd = bool(job_description and job_description.strip())

        # Step 1: Extract all projects from both 'projects' and 'experience' sections
        all_projects = self.extract_all_projects(safe_resume)
        
        final_projects_to_enhance = []
        if has_jd and all_projects:
            # Step 2: Score all projects based on relevance to the job description
            st.write(f"Scoring {len(all_projects)} potential projects for relevance...")
            scored_projects = []
            progress_bar_score = st.progress(0)
            for i, project in enumerate(all_projects):
                score = self.score_project_relevance(project, job_keywords)
                scored_projects.append((project, score))
                progress_bar_score.progress((i + 1) / len(all_projects))
            progress_bar_score.empty()

            # Sort projects by score in descending order
            scored_projects.sort(key=lambda x: x[1], reverse=True)
            
            # Step 3: Select the top 6-8 projects for the resume
            final_projects_to_enhance = [p for p, s in scored_projects[:8]]
        else:
            # If no JD, just take the first 6 projects found
            final_projects_to_enhance = all_projects[:6]

        # Step 4: Enhance the description for each of the selected projects
        enhanced_projects = []
        if final_projects_to_enhance:
            st.write(f"Enhancing {len(final_projects_to_enhance)} selected projects...")
            progress_bar_enhance = st.progress(0)
            for i, project in enumerate(final_projects_to_enhance):
                enhanced_project = self.enhance_project_description(project, job_keywords if has_jd else set())
                enhanced_projects.append(enhanced_project)
                # Adding a small delay to avoid hitting API rate limits if calls are too fast
                time.sleep(0.5) 
                progress_bar_enhance.progress((i + 1) / len(final_projects_to_enhance))
            progress_bar_enhance.empty()

        safe_resume['projects'] = enhanced_projects

        # Step 5: Generate title, summary, and skills list
        if has_jd:
            safe_resume["title"] = self.generate_professional_title(safe_resume, job_keywords, job_description)
            safe_resume["summary"] = self.generate_professional_summary(safe_resume, job_keywords)
        
        original_skills = safe_resume.get("skills", [])
        safe_resume["skills"] = self.optimize_skills_list(original_skills, job_keywords if has_jd else set())

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
    
    def expand_project_description(self, project: Dict, job_keywords: Set[str], is_relevant: bool = True) -> str:
        """Expand a project description based on its title and minimal description."""
        # Adjust the prompt based on whether project is relevant to job keywords
        if is_relevant:
            keyword_instruction = "7. CRITICAL PRIORITY: This project is highly relevant to the job. Make sure to prominently highlight and emphasize any skills, technologies, or experiences that match the provided job keywords.\n8. If the project uses any of the job keywords, make sure to explicitly mention them in the description and show strong alignment with job requirements."
        else:
            keyword_instruction = "7. IMPORTANT: While this project may not directly match the job keywords, highlight transferable skills and demonstrate impact.\n8. Focus on general technical competencies, problem-solving abilities, and quantifiable achievements that show professional growth."
        
        system_prompt = f"""
You are a professional technical resume writer. Your task is to convert raw project data into a concise, impactful, and metric-driven project description suitable for a resume.

Follow these rules:

1. Output 4 to 7 bullet points, each 1–2 lines long maximum. Don't write extra, be to the point and concise.
2. The first line compulsorily has to be a professionally retailored title of the project, do this for every project and make sure it has only first letter of each word capitalised.
    -> If its a work experience, make sure not to include company name in the title and give a generic title to the specific project.
3. Each bullet point must start with a strong past-tense action verb (Engineered, Built, Automated, Designed, Achieved, etc.).
4. The bullet points should:
 -> summarize what you built, what technologies you used, and why it mattered — no fluff, no LLM mentions unless critical.
 -> describe any UX, responsiveness, optimization, version control, or frontend/backend implementation details.
 -> include specific results using clear metrics — e.g., "93% coverage," "reduced latency by 40%," "automated 80% of manual work."
 -> explain the business or user impact — how it helped the client, user, or team.
5. Keep each bullet short and crisp — aim for clarity, not verbosity.
6. Format each bullet point with a bullet symbol (•) at the beginning.
{keyword_instruction}

You must NOT generate a paragraph — return the project title on the first line, followed by 4 to 7 bullet points starting with (•).

Example format:
Enhanced E-Commerce Platform
• Engineered responsive web application using React, Node.js, and MongoDB, serving 10,000+ daily users
• Implemented payment gateway integration with Stripe API, increasing conversion rates by 25%
• Optimized database queries and API responses, reducing page load time by 40%
• Built automated testing suite with Jest and Cypress, achieving 95% code coverage
"""

        user_prompt = f"""
Project Title: {project['title']}
Raw Description: {project['description']}
Technologies: {', '.join(project.get('technologies', []))}
Job Keywords to Highlight: {', '.join(job_keywords)}
Project Relevance: {'HIGH - This project is directly relevant to the job requirements' if is_relevant else 'MODERATE - This project shows transferable skills and general competencies'}

Using the system instructions, create a professional project title followed by 4 to 7 clear, action-driven bullet points (starting with •) that summarize the project. {'Emphasize strong alignment with job keywords and requirements.' if is_relevant else 'Focus on transferable skills, technical competencies, and quantifiable impact.'}

Required format:
[Professional Project Title]
• [Action verb] [description with technologies and impact]
• [Action verb] [description with metrics/results]
• [Action verb] [description {'highlighting job keyword alignment' if is_relevant else 'showcasing transferable skills'}]
• [Additional bullet points as needed]
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
- Rewrite the 'summary' field to be a concise, job-specific summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords
- Add or update a 'title' field in the resume JSON, inferring a proper, professional job title for the candidate based on the job description and their experience/skills. The title should be a realistic job title (e.g., 'Frontend Developer', 'Data Scientist', 'Project Manager'), not just a single keyword or technology. Do not leave the title blank. If unsure, use the most relevant job title from the job description.
- Extract projects from BOTH 'projects' and 'experience' sections - work experience descriptions containing projects or achievements should be converted to project format and included in the projects section.
- Do NOT add, invent, or hallucinate any new skills, projects, or summary content.
- Do NOT change or add any other fields except 'title', 'projects', and 'summary'. KEEP 'skills' exactly as provided in the original resume (post-processing will handle prioritization and limiting to 22 skills).
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
2. EXTRACT ALL PROJECTS from BOTH 'projects' and 'experience' sections - Convert work experience descriptions into project format and include them along with existing projects. Include ALL projects regardless of job keyword relevance (post-processing will handle prioritization and enhancement).
3. Rewrite the 'summary' field to be a 2-4 sentence summary that highlights the candidate's fit for the job, using only information from the resume and the job keywords.
4. Do NOT add or invent any new skills, projects, or summary content - only extract and restructure existing information.
5. Return the complete resume in the exact same JSON format, with the original skills kept as provided, ALL projects (from both sources), and new summary.
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
            
            # IMPORTANT: Prioritize and limit skills to maximum 22, focusing on job relevance
            original_skills = list(safe_resume.get("skills", []))
            job_keywords_list = list(job_keywords)
            
            # Convert to lowercase for case-insensitive comparison
            job_keywords_lower = {k.lower() for k in job_keywords_list}
            original_skills_lower = {s.lower(): s for s in original_skills}  # Keep original casing
            
            # Priority 1: Original skills that match job keywords (highest priority)
            matching_skills = []
            for keyword_lower in job_keywords_lower:
                if keyword_lower in original_skills_lower:
                    matching_skills.append(original_skills_lower[keyword_lower])
            
            # Priority 2: Job keywords that weren't in original skills
            missing_keywords = []
            for keyword in job_keywords_list:
                if keyword.lower() not in original_skills_lower:
                    missing_keywords.append(keyword)
            
            # Priority 3: Remaining original skills (non-matching ones)
            remaining_skills = []
            for skill in original_skills:
                if skill.lower() not in job_keywords_lower:
                    remaining_skills.append(skill)
            
            # Combine skills with priority order and limit to 22
            prioritized_skills = matching_skills + missing_keywords + remaining_skills
            
            # Limit to maximum 22 skills
            final_skills = prioritized_skills[:22]
            
            retailored_resume["skills"] = final_skills
            
            # --- Enhanced logic: Include ALL projects (relevant + non-relevant) with prioritization ---
            # Combine all projects with relevant ones first, then non-relevant ones
            all_projects = relevant_projects + non_relevant_projects
            
            # If we have too many projects, prioritize relevant ones but include some non-relevant
            if len(all_projects) > 8:  # Limit to reasonable number for resume
                # Take all relevant projects + top non-relevant projects to make total of 8
                max_non_relevant = max(0, 8 - len(relevant_projects))
                final_projects = relevant_projects + non_relevant_projects[:max_non_relevant]
            else:
                final_projects = all_projects
            
            # Enhance ALL projects with job-aligned descriptions
            enhanced_projects = []
            for project in final_projects:
                # Mark if project is relevant for enhanced processing
                is_relevant = project in relevant_projects
                
                # Enhanced description for all projects, with extra focus on relevant ones
                enhanced_description = self.expand_project_description(project, job_keywords, is_relevant)
                
                # Create enhanced project
                enhanced_project = project.copy()
                enhanced_project["description"] = enhanced_description
                enhanced_projects.append(enhanced_project)
                
                time.sleep(1)  # Rate limiting for API calls
            
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