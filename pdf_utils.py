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
            if not text or not keywords:
                return text
            
            # Convert to string if not already
            text = str(text)
            
            # Sort keywords by length (longest first) to avoid partial matches
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            
            for kw in sorted_keywords:
                if len(kw.strip()) < 2:  # Skip very short keywords
                    continue
                    
                # Escape special regex characters
                escaped_kw = re.escape(kw.strip())
                
                # Use regex for whole word, case-insensitive match
                # Avoid bolding if already bolded
                pattern = r'(?i)(?<!<strong>)\b({})\b(?!</strong>)'.format(escaped_kw)
                text = re.sub(pattern, r'<strong>\1</strong>', text)
                
                # For technical terms that might be part of compound words
                # Only if the keyword is longer than 3 characters
                if len(kw.strip()) > 3:
                    pattern_substring = r'(?i)(?<!<strong>)({})(?!</strong>)'.format(escaped_kw)
                    # Only apply if not already bolded and not part of a larger bolded section
                    if f'<strong>{kw.lower()}</strong>' not in text.lower():
                        text = re.sub(pattern_substring, r'<strong>\1</strong>', text)
            
            return text

        # Get keywords as a set (if provided)
        keywords_set = set(keywords) if keywords else set()

        # Automatically convert project descriptions to bullet points
        projects = data.get('projects', [])
        for project in projects:
            if 'description' in project and isinstance(project['description'], str):
                desc = paragraph_to_bullets(project['description'])
                project['description'] = bold_keywords(desc, keywords_set)
            if 'title' in project and isinstance(project['title'], str):
                project['title'] = bold_keywords(project['title'], keywords_set)

        # Bold keywords in summary
        if 'summary' in data and isinstance(data['summary'], str):
            data['summary'] = bold_keywords(data['summary'], keywords_set)

        # Bold keywords in title
        if 'title' in data and isinstance(data['title'], str):
            data['title'] = bold_keywords(data['title'], keywords_set)

        # Bold keywords in skills
        if 'skills' in data and isinstance(data['skills'], list):
            data['skills'] = [bold_keywords(skill, keywords_set) for skill in data['skills']]

        # Bold keywords in education
        if 'education' in data and isinstance(data['education'], list):
            for edu in data['education']:
                if isinstance(edu, dict):
                    if 'degree' in edu and edu['degree']:
                        edu['degree'] = bold_keywords(edu['degree'], keywords_set)
                    if 'institution' in edu and edu['institution']:
                        edu['institution'] = bold_keywords(edu['institution'], keywords_set)
                    if 'year' in edu and edu['year']:
                        edu['year'] = bold_keywords(str(edu['year']), keywords_set)

        # Bold keywords in certifications
        if 'certifications' in data and isinstance(data['certifications'], list):
            for i, cert in enumerate(data['certifications']):
                if isinstance(cert, dict):
                    if 'title' in cert and cert['title']:
                        cert['title'] = bold_keywords(cert['title'], keywords_set)
                    if 'issuer' in cert and cert['issuer']:
                        cert['issuer'] = bold_keywords(cert['issuer'], keywords_set)
                    if 'year' in cert and cert['year']:
                        cert['year'] = bold_keywords(str(cert['year']), keywords_set)
                elif isinstance(cert, str):
                    data['certifications'][i] = bold_keywords(cert, keywords_set)

        # --- New logic for section-aware multi-page handling ---
        continuation_template = env.get_template('templates/template_continuation.html')

        # Estimate how many items fit in the left and right columns per page
        LEFT_COL_MAX = 28  # Total items per page in left column
        RIGHT_COL_MAX = 2  # projects per page

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
            sections = ['skills', 'certifications', 'education']
            pages = []
            current_page = {'skills': [], 'certifications': [], 'education': []}
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
                    current_page = {'skills': [], 'certifications': [], 'education': []}
                    current_page_size = 0
                
                # Add entire section to current page
                current_page[section_name] = section_data
                current_page_size += section_size
            
            # Add the last page if it has content
            if any(current_page.values()):
                pages.append(current_page)
            
            return pages

        # Prepare left column content
        left_column = {
            'skills': data.get('skills', []),
            'certifications': data.get('certifications', []),
            'education': data.get('education', [])
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
                size += len(bullets)
            return size
        
        def distribute_projects_to_pages(projects, max_space_per_page=12):
            """Distribute projects across pages based on content size"""
            if not projects:
                return [[]]
                
            pages = []
            current_page = []
            current_page_size = 0
            
            for project in projects:
                project_size = estimate_project_size(project)
                
                # If this project can't fit on current page, start a new page
                if current_page_size + project_size > max_space_per_page and current_page:
                    pages.append(current_page)
                    current_page = []
                    current_page_size = 0
                
                # Add project to current page
                current_page.append(project)
                current_page_size += project_size
            
            # Add the last page if it has content
            if current_page:
                pages.append(current_page)
            
            return pages if pages else [[]]
        
        projects = data.get('projects', [])
        right_chunks = distribute_projects_to_pages(projects, max_space_per_page=12)

        # Ensure we have at least one page for each column
        if not left_pages:
            left_pages = [{'skills': [], 'certifications': [], 'education': []}]
        if not right_chunks:
            right_chunks = [[]]

        # First page: render with first page of left column and first chunk of right column
        data_copy = copy.deepcopy(data)
        first_left_page = left_pages[0]
        data_copy['skills'] = first_left_page['skills']
        data_copy['certifications'] = first_left_page['certifications']
        data_copy['education'] = first_left_page['education']
        data_copy['projects'] = right_chunks[0]
        
        html_pages = [template.render(
            cv=data_copy,
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
            
            html_pages.append(continuation_template.render(
                left_column=left_col,
                right_column=right_col,
                section_headings=section_headings,
                project_index_offset=project_index_offset,
                font_size=font_size,
                left_logo=f"data:image/png;base64,{left_logo_b64}",
                right_logo=f"data:image/png;base64,{right_logo_b64}"
            ))
            project_index_offset += len(right_col)

        # Combine all HTMLs
        full_html = ''.join(html_pages)
        pdf_file = io.BytesIO()
        HTML(string=full_html).write_pdf(pdf_file)
        return pdf_file, full_html

 
