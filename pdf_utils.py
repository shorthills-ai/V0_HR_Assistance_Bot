import base64
import io
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import fitz  # PyMuPDF
import copy
import re

class PDFUtils:
    @staticmethod
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    @staticmethod
    def get_base64_pdf(pdf_file):
        pdf_file.seek(0)
        return base64.b64encode(pdf_file.read()).decode("utf-8")

    @staticmethod
    def analyze_space_usage(data, max_space_per_page=25):
        """
        Analyze space usage for projects without generating the full PDF.
        Returns detailed information about how space will be distributed.
        
        Args:
            data: Resume data dictionary
            max_space_per_page: Maximum space units per page for projects (default: 18)
            
        Returns:
            dict: Contains space analysis information including:
                - total_projects: Number of projects
                - total_estimated_space: Total space needed for all projects
                - estimated_pages: Number of pages needed
                - space_per_page: List of space used per page
                - remaining_space_per_page: List of remaining space per page
                - project_details: List of project space breakdowns
        """
        def estimate_project_size(project):
            """Estimate how much space a project will take"""
            size = 1  # Base size for project title
            if project.get('description'):
                bullets = project['description'].split('\n')
                bullets = [b.strip() for b in bullets if b.strip()]
                size += int(len(bullets) * 0.8)
            return size
        
        projects = data.get('projects', [])
        
        if not projects:
            return {
                'total_projects': 0,
                'total_estimated_space': 0,
                'estimated_pages': 1,
                'space_per_page': [0],
                'remaining_space_per_page': [max_space_per_page],
                'project_details': []
            }
        
        # Analyze each project
        project_details = []
        total_estimated_space = 0
        
        for i, project in enumerate(projects):
            project_size = estimate_project_size(project)
            bullet_count = 0
            if project.get('description'):
                bullets = project['description'].split('\n')
                bullet_count = len([b.strip() for b in bullets if b.strip()])
            
            project_info = {
                'index': i + 1,
                'title': project.get('title', 'Untitled'),
                'title_space': 1,
                'bullet_count': bullet_count,
                'total_space': project_size
            }
            project_details.append(project_info)
            total_estimated_space += project_size
        
        # Distribute projects to pages
        pages = []
        current_page = []
        current_page_size = 0
        space_per_page = []
        remaining_space_per_page = []
        
        usable_space_per_page = max_space_per_page - 2
        
        for project in projects:
            project_size = estimate_project_size(project)
            
            # If this project can't fit on current page, start a new page
            if current_page_size + project_size > usable_space_per_page and current_page:
                space_per_page.append(current_page_size)
                remaining_space_per_page.append(usable_space_per_page - current_page_size)
                pages.append(current_page)
                current_page = []
                current_page_size = 0
            
            # Add project to current page
            current_page.append(project)
            current_page_size += project_size
        
        # Add the last page if it has content
        if current_page:
            space_per_page.append(current_page_size)
            remaining_space_per_page.append(usable_space_per_page - current_page_size)
            pages.append(current_page)
        
        estimated_pages = len(pages) if pages else 1
        
        return {
            'total_projects': len(projects),
            'total_estimated_space': total_estimated_space,
            'estimated_pages': estimated_pages,
            'space_per_page': space_per_page,
            'remaining_space_per_page': remaining_space_per_page,
            'project_details': project_details,
            'max_space_per_page': max_space_per_page
        }

    @staticmethod
    def generate_pdf(data, template_path="templates/template.html",
                     bg_path="templates/bg.png",
                     left_logo_path="templates/left_logo_small.png",
                     right_logo_path="templates/right_logo_small.png",
                     keywords=None):
        """
        Generate a PDF resume from the provided data and template.
        If 'keywords' is provided (as a set or list of strings), all occurrences of those keywords
        in the resume (summary, project descriptions, skills, certifications, education, project titles)
        will be automatically bolded.

        Example usage:
            keywords = JobDescriptionAnalyzer().extract_keywords(job_description)["keywords"]
            PDFUtils.generate_pdf(data, keywords=keywords)
        """
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template(template_path)

        bg_image = PDFUtils.get_base64_image(bg_path)
        left_logo_b64 = PDFUtils.get_base64_image(left_logo_path)
        right_logo_b64 = PDFUtils.get_base64_image(right_logo_path)

        def paragraph_to_bullets(paragraph):
            if not isinstance(paragraph, str):
                return paragraph
            sentences = re.split(r'(?<=[.!?]) +', paragraph.strip())
            bullets = [s.strip() for s in sentences if s.strip()]
            return '\n'.join(bullets)

        def bold_keywords(text, keywords):
            """Bold keywords in text with improved matching"""
            if not text or not keywords:
                return text
            
            # Convert text to string if it's not already
            text = str(text)
            original_text = text  # Keep for debugging
            
            # Convert keywords to list and filter out empty ones
            if isinstance(keywords, set):
                keyword_list = list(keywords)
            elif isinstance(keywords, (list, tuple)):
                keyword_list = list(keywords)
            else:
                return text
            
            # Filter and clean keywords
            valid_keywords = []
            for kw in keyword_list:
                if kw and isinstance(kw, str) and len(kw.strip()) >= 2:
                    valid_keywords.append(kw.strip())
            
            if not valid_keywords:
                return text
            
            # Sort keywords by length (longest first) to avoid partial matches
            valid_keywords.sort(key=len, reverse=True)
            
            # Apply bolding for each keyword
            for keyword in valid_keywords:
                # Escape special regex characters
                escaped_keyword = re.escape(keyword)
                
                # Create multiple patterns to catch different cases
                patterns = [
                    # Exact match with word boundaries (case insensitive)
                    r'(?i)(?<!<strong>)\b(' + escaped_keyword + r')\b(?![^<]*</strong>)',
                    # Match within parentheses like "(DRDO)"
                    r'(?i)(?<!<strong>)\(([^)]*' + escaped_keyword + r'[^)]*)\)(?![^<]*</strong>)',
                    # Match acronyms and abbreviations
                    r'(?i)(?<!<strong>)(' + escaped_keyword + r')(?![^<]*</strong>)'
                ]
                
                for pattern in patterns:
                    if re.search(pattern, text):
                        print(f"DEBUG: Found keyword '{keyword}' in text: {text[:50]}...")
                        text = re.sub(pattern, r'<strong>\1</strong>', text)
                        break
            
            # Debug output if text changed
            if text != original_text:
                print(f"DEBUG: Text transformed from: {original_text}")
                print(f"DEBUG: Text transformed to: {text}")
            
            return text

        # Create a deep copy to avoid modifying original data
        data_copy = copy.deepcopy(data)
        
        # Convert keywords to proper format
        if keywords:
            if isinstance(keywords, dict) and 'keywords' in keywords:
                keywords_to_use = keywords['keywords']
            else:
                keywords_to_use = keywords
        else:
            keywords_to_use = set()

        # Debug: Print keywords being used (remove in production)
        print(f"DEBUG: Keywords being used for bolding: {keywords_to_use}")
        print(f"DEBUG: Type of keywords: {type(keywords_to_use)}")

        # Bold keywords in certifications - ENHANCED with better debugging
        if 'certifications' in data_copy and isinstance(data_copy['certifications'], list):
            print(f"DEBUG: Processing {len(data_copy['certifications'])} certifications")
            for i, cert in enumerate(data_copy['certifications']):
                print(f"DEBUG: Processing certification {i}: {cert}")
                if isinstance(cert, dict):
                    # Bold certification title
                    if 'title' in cert and cert['title']:
                        print(f"DEBUG: Bolding certification title: {cert['title']}")
                        original_title = cert['title']
                        cert['title'] = bold_keywords(cert['title'], keywords_to_use)
                        if original_title != cert['title']:
                            print(f"DEBUG: Certification title bolded - '{original_title}' -> '{cert['title']}'")
                    
                    # Bold certification issuer/organization
                    if 'issuer' in cert and cert['issuer']:
                        print(f"DEBUG: Bolding certification issuer: {cert['issuer']}")
                        original_issuer = cert['issuer']
                        cert['issuer'] = bold_keywords(cert['issuer'], keywords_to_use)
                        if original_issuer != cert['issuer']:
                            print(f"DEBUG: Certification issuer bolded - '{original_issuer}' -> '{cert['issuer']}'")
                        else:
                            print(f"DEBUG: No keywords found in issuer: {cert['issuer']}")
                    
                    # Also check for 'organization' field if it exists
                    if 'organization' in cert and cert['organization']:
                        print(f"DEBUG: Bolding certification organization: {cert['organization']}")
                        original_org = cert['organization']
                        cert['organization'] = bold_keywords(cert['organization'], keywords_to_use)
                        if original_org != cert['organization']:
                            print(f"DEBUG: Certification organization bolded - '{original_org}' -> '{cert['organization']}'")
                
                elif isinstance(cert, str) and cert:
                    # Handle string certifications
                    print(f"DEBUG: Bolding string certification: {cert}")
                    original_cert = cert
                    data_copy['certifications'][i] = bold_keywords(cert, keywords_to_use)
                    if original_cert != data_copy['certifications'][i]:
                        print(f"DEBUG: String certification bolded - '{original_cert}' -> '{data_copy['certifications'][i]}'")
                    else:
                        print(f"DEBUG: No keywords found in string certification: {cert}")

        # Debug: Print keywords being used (remove in production)
        print(f"DEBUG: Keywords being used for bolding: {keywords_to_use}")

        # Bold keywords in summary
        if 'summary' in data_copy and data_copy['summary']:
            original_summary = data_copy['summary']
            data_copy['summary'] = bold_keywords(original_summary, keywords_to_use)
            print(f"DEBUG: Summary bolding - Original: {original_summary[:100]}...")
            print(f"DEBUG: Summary bolding - Modified: {data_copy['summary'][:100]}...")

        # Bold keywords in skills
        if 'skills' in data_copy and isinstance(data_copy['skills'], list):
            for i, skill in enumerate(data_copy['skills']):
                if skill:
                    original_skill = skill
                    data_copy['skills'][i] = bold_keywords(skill, keywords_to_use)
                    if original_skill != data_copy['skills'][i]:
                        print(f"DEBUG: Skill bolded - '{original_skill}' -> '{data_copy['skills'][i]}'")

        # Bold keywords in projects
        if 'projects' in data_copy and isinstance(data_copy['projects'], list):
            for project in data_copy['projects']:
                # Bold project title
                if 'title' in project and project['title']:
                    original_title = project['title']
                    project['title'] = bold_keywords(project['title'], keywords_to_use)
                    if original_title != project['title']:
                        print(f"DEBUG: Project title bolded - '{original_title}' -> '{project['title']}'")
                
                # Bold project description
                if 'description' in project and project['description']:
                    # First convert to bullets
                    desc = paragraph_to_bullets(project['description'])
                    # Then bold keywords
                    original_desc = desc
                    project['description'] = bold_keywords(desc, keywords_to_use)
                    if original_desc != project['description']:
                        print(f"DEBUG: Project description bolded")

        # Bold keywords in education
        if 'education' in data_copy and isinstance(data_copy['education'], list):
            for edu in data_copy['education']:
                if isinstance(edu, dict):
                    if 'degree' in edu and edu['degree']:
                        original_degree = edu['degree']
                        edu['degree'] = bold_keywords(edu['degree'], keywords_to_use)
                        if original_degree != edu['degree']:
                            print(f"DEBUG: Education degree bolded - '{original_degree}' -> '{edu['degree']}'")
                    if 'institution' in edu and edu['institution']:
                        original_institution = edu['institution']
                        edu['institution'] = bold_keywords(edu['institution'], keywords_to_use)
                        if original_institution != edu['institution']:
                            print(f"DEBUG: Education institution bolded - '{original_institution}' -> '{edu['institution']}'")


        # --- Section-aware multi-page handling (using modified data) ---
        continuation_template = env.get_template('templates/template_continuation.html')

        # Estimate how many items fit in the left and right columns per page
        LEFT_COL_MAX = 28  # Total items per page in left column


        font_size = 13  # Default font size for all pages

        def estimate_section_size(section_data):
            """Estimate how many 'item slots' a section will take"""
            if not section_data:
                return 0
            # Add 1 for section title + number of items
            return 1 + len(section_data)

        def can_fit_section(current_size, section_size, max_size):
            """Check if a section can fit in remaining space"""
            return current_size + section_size <= max_size

        def distribute_sections_to_pages(left_column_data, max_items_per_page):
            """Distribute sections across pages, keeping each section intact"""
            sections = ['skills', 'education', 'certifications']
            pages = []
            current_page = {'skills': [], 'education': [], 'certifications': []}
            current_page_size = 0
            
            for section_name in sections:
                section_data = left_column_data.get(section_name, [])
                if not section_data:
                    continue
                    
                section_size = estimate_section_size(section_data)
                
                # If this section can't fit on current page, start a new page
                if not can_fit_section(current_page_size, section_size, max_items_per_page):
                    # Save current page if it has content
                    if any(current_page.values()):
                        pages.append(current_page)
                    # Start new page
                    current_page = {'skills': [], 'education': [], 'certifications': []}
                    current_page_size = 0
                
                # Add entire section to current page
                current_page[section_name] = section_data
                current_page_size += section_size
            
            # Add the last page if it has content
            if any(current_page.values()):
                pages.append(current_page)
            
            return pages

        # Prepare left column content using the keyword-bolded data
        left_column = {
            'skills': data_copy.get('skills', []),
            'certifications': data_copy.get('certifications', []),
            'education': data_copy.get('education', [])
        }

        # Distribute sections across pages
        left_pages = distribute_sections_to_pages(left_column, LEFT_COL_MAX)
        
        # Handle right column (projects) - content-aware pagination
        def estimate_project_size(project):
            """Estimate how much space a project will take"""
            size = 1  # Base size for project title
            if project.get('description'):
                # Count bullet points in description
                bullets = project['description'].split('\n')
                bullets = [b.strip() for b in bullets if b.strip()]
                size += int(len(bullets) * 0.5)
            return size
        
        def analyze_project_space_usage(projects, max_space_per_page=25):
            """Analyze and report detailed space usage for projects"""
            print(f"\n=== DETAILED PROJECT SPACE ANALYSIS ===")
            print(f"Max space per page: {max_space_per_page}")
            print(f"Total projects: {len(projects)}")
            
            total_estimated_space = 0
            for i, project in enumerate(projects):
                project_size = estimate_project_size(project)
                total_estimated_space += project_size
                bullet_count = 0
                if project.get('description'):
                    bullets = project['description'].split('\n')
                    bullet_count = len([b.strip() for b in bullets if b.strip()])
                
                print(f"  Project {i+1}: '{project.get('title', 'Untitled')[:50]}...'")
                print(f"    - Title space: 1")
                print(f"    - Bullet points: {bullet_count}")
                print(f"    - Total estimated space: {project_size}")
            
            estimated_pages = (total_estimated_space + max_space_per_page - 1) // max_space_per_page  # Ceiling division
            print(f"\nTotal estimated space needed: {total_estimated_space}")
            print(f"Estimated pages needed: {estimated_pages}")
            print(f"Average space per page: {total_estimated_space / estimated_pages if estimated_pages > 0 else 0:.1f}")
            print(f"========================================\n")
            
            return total_estimated_space, estimated_pages
        
        def distribute_projects_to_pages(projects, max_space_per_page_first, max_space_per_page_rest):
            if not projects:
                return [[]], [max_space_per_page_first - 2], [[]]

            pages = []
            current_page = []
            current_page_size = 0
            space_remaining_per_page = []
            project_numbers_per_page = []
            current_page_numbers = []
            project_counter = 0
            page_idx = 0

            i = 0
            while i < len(projects):
                project = projects[i]
                desc_lines = [l.strip() for l in project.get('description', '').split('\n') if l.strip()]
                idx = 0
                first_chunk = True
                while idx < len(desc_lines) or (idx == 0 and len(desc_lines) == 0):
                    # Choose max space for this page
                    max_space_per_page = max_space_per_page_first if page_idx == 0 else max_space_per_page_rest
                    usable_space_per_page = max_space_per_page - 2
                    remaining_space = usable_space_per_page - current_page_size
                    if first_chunk:
                        lines_for_this_page = max(remaining_space - 1, 0)
                        if lines_for_this_page <= 0:
                            if current_page:
                                space_remaining_per_page.append(usable_space_per_page - current_page_size)
                                pages.append(current_page)
                                project_numbers_per_page.append(current_page_numbers)
                                page_idx += 1
                            current_page = []
                            current_page_numbers = []
                            current_page_size = 0
                            max_space_per_page = max_space_per_page_first if page_idx == 0 else max_space_per_page_rest
                            usable_space_per_page = max_space_per_page - 2
                            remaining_space = usable_space_per_page
                            lines_for_this_page = max(remaining_space - 1, 0)
                        chunk_lines = desc_lines[idx:idx+lines_for_this_page]
                        split_proj = dict(project)
                        split_proj['description'] = '\n'.join(chunk_lines)
                        current_page.append(split_proj)
                        project_counter += 1
                        current_page_numbers.append(project_counter)
                        current_page_size += 1 + len(chunk_lines)
                        idx += len(chunk_lines)
                        first_chunk = False
                        if len(chunk_lines) == 0:
                            break
                    else:
                        lines_for_this_page = remaining_space
                        if lines_for_this_page <= 0:
                            if current_page:
                                space_remaining_per_page.append(usable_space_per_page - current_page_size)
                                pages.append(current_page)
                                project_numbers_per_page.append(current_page_numbers)
                                page_idx += 1
                            current_page = []
                            current_page_numbers = []
                            current_page_size = 0
                            max_space_per_page = max_space_per_page_first if page_idx == 0 else max_space_per_page_rest
                            usable_space_per_page = max_space_per_page - 2
                            remaining_space = usable_space_per_page
                            lines_for_this_page = remaining_space
                        chunk_lines = desc_lines[idx:idx+lines_for_this_page]
                        split_proj = dict(project)
                        split_proj['description'] = '\n'.join(chunk_lines)
                        split_proj['title'] = ''
                        if 'number' in split_proj:
                            del split_proj['number']
                        current_page.append(split_proj)
                        current_page_numbers.append(None)
                        current_page_size += len(chunk_lines)
                        idx += len(chunk_lines)
                    if idx >= len(desc_lines):
                        break
                    if current_page_size >= usable_space_per_page:
                        space_remaining_per_page.append(usable_space_per_page - current_page_size)
                        pages.append(current_page)
                        project_numbers_per_page.append(current_page_numbers)
                        page_idx += 1
                        current_page = []
                        current_page_numbers = []
                        current_page_size = 0
                if len(desc_lines) == 0:
                    max_space_per_page = max_space_per_page_first if page_idx == 0 else max_space_per_page_rest
                    usable_space_per_page = max_space_per_page - 2
                    if current_page_size + 1 > usable_space_per_page:
                        space_remaining_per_page.append(usable_space_per_page - current_page_size)
                        pages.append(current_page)
                        project_numbers_per_page.append(current_page_numbers)
                        page_idx += 1
                        current_page = []
                        current_page_numbers = []
                        current_page_size = 0
                    split_proj = dict(project)
                    split_proj['description'] = ''
                    current_page.append(split_proj)
                    project_counter += 1
                    current_page_numbers.append(project_counter)
                    current_page_size += 1
                i += 1
            if current_page:
                max_space_per_page = max_space_per_page_first if page_idx == 0 else max_space_per_page_rest
                usable_space_per_page = max_space_per_page - 2
                space_remaining_per_page.append(usable_space_per_page - current_page_size)
                pages.append(current_page)
                project_numbers_per_page.append(current_page_numbers)
            if not pages:
                max_space_per_page = max_space_per_page_first
                usable_space_per_page = max_space_per_page - 2
                pages = [[]]
                space_remaining = [usable_space_per_page]
                project_numbers_per_page = [[]]
            return pages, space_remaining_per_page, project_numbers_per_page

        projects = data_copy.get('projects', [])
        # Use different max space for first and subsequent pages
        max_space_per_page_first = 25
        max_space_per_page_rest = 28
        right_chunks, space_remaining, project_numbers_per_page = distribute_projects_to_pages(projects, max_space_per_page_first, max_space_per_page_rest)

        # --- Fix: Pad project_numbers_per_page and space_remaining to match right_chunks ---
        while len(project_numbers_per_page) < len(right_chunks):
            project_numbers_per_page.append([])
        while len(space_remaining) < len(right_chunks):
            space_remaining.append(15)

        # Print space utilization summary
        print(f"\n=== PROJECT SPACE UTILIZATION SUMMARY ===")
        for i, (chunk, remaining) in enumerate(zip(right_chunks, space_remaining)):
            used_space = 12 - remaining  # max_space_per_page - remaining
            utilization_percent = (used_space / 12) * 100
            print(f"Page {i+1}: {len(chunk)} projects, {used_space}/12 space used ({utilization_percent:.1f}%), {remaining} space remaining")
        print(f"==========================================\n")

        # Ensure we have at least one page for each column
        if not left_pages:
            left_pages = [{'skills': [], 'certifications': [], 'education': []}]
        if not right_chunks:
            right_chunks = [[]]
            space_remaining = [12]  # Full space available if no projects

        # Debug: Print lengths before rendering
        print(f"DEBUG: right_chunks: {len(right_chunks)}, project_numbers_per_page: {len(project_numbers_per_page)}, space_remaining: {len(space_remaining)}")
        if len(right_chunks) != len(project_numbers_per_page):
            print(f"WARNING: right_chunks and project_numbers_per_page length mismatch!")
        if len(right_chunks) != len(space_remaining):
            print(f"WARNING: right_chunks and space_remaining length mismatch!")

        # First page: render with first page of left column and first chunk of right column
        first_page_data = copy.deepcopy(data_copy)
        first_left_page = left_pages[0]
        first_page_data['skills'] = first_left_page['skills']
        first_page_data['certifications'] = first_left_page['certifications']
        first_page_data['education'] = first_left_page['education']
        first_page_data['projects'] = right_chunks[0]
        
        html_pages = [template.render(
            cv=first_page_data,
            bg_image=f"data:image/png;base64,{bg_image}",
            left_logo=f"data:image/png;base64,{left_logo_b64}",
            right_logo=f"data:image/png;base64,{right_logo_b64}",
            font_size=font_size
        )]

        # Track which sections appeared on first page for continuation page headings
        first_page_sections = {
            'skills': len(first_left_page['skills']) > 0,
            'certifications': len(first_left_page['certifications']) > 0,
            'education': len(first_left_page['education']) > 0
        }

        # Render continuation pages
        project_index_offset = len(right_chunks[0])
        max_pages = max(len(left_pages), len(right_chunks))
        
        for page_idx in range(1, max_pages):
            # Get left column content for this page
            left_col = {'skills': [], 'certifications': [], 'education': []}
            section_headings = {'skills': False, 'certifications': False, 'education': False}
            
            if page_idx < len(left_pages):
                left_col = left_pages[page_idx]
                # Determine which section headings to show
                # Show heading only if the entire section moved to this page (not on first page)
                for section in ['skills', 'certifications', 'education']:
                    if left_col[section] and not first_page_sections[section]:
                        section_headings[section] = True
            
            # Get right column content for this page
            right_col = []
            if page_idx < len(right_chunks):
                right_col = right_chunks[page_idx]
            # Safe fallback for project numbers
            if page_idx < len(project_numbers_per_page):
                right_project_numbers = project_numbers_per_page[page_idx]
            else:
                right_project_numbers = []
            
            html_pages.append(continuation_template.render(
                left_column=left_col,
                right_column=right_col,
                section_headings=section_headings,
                project_index_offset=project_index_offset,
                font_size=font_size,
                left_logo=f"data:image/png;base64,{left_logo_b64}",
                right_logo=f"data:image/png;base64,{right_logo_b64}",
                right_project_numbers=right_project_numbers
            ))
            project_index_offset += len(right_col)

        # Combine all HTMLs
        full_html = ''.join(html_pages)
        pdf_file = io.BytesIO()
        HTML(string=full_html).write_pdf(pdf_file)
        return pdf_file, full_html