import base64
import io
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import fitz  # PyMuPDF
import copy
import re
import docx
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.shared import qn
from docx.enum.dml import MSO_THEME_COLOR_INDEX

class DocxUtils:
    @staticmethod
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    @staticmethod
    def get_base64_pdf(pdf_file):
        pdf_file.seek(0)
        return base64.b64encode(pdf_file.read()).decode("utf-8")

    @staticmethod
    def add_image_to_header(doc, image_path, position='left'):
        """Add image to document header"""
        try:
            section = doc.sections[0]
            header = section.header
            header_para = header.paragraphs[0]
            header_para.alignment = WD_ALIGN_PARAGRAPH.LEFT if position == 'left' else WD_ALIGN_PARAGRAPH.RIGHT
            
            # Clear existing content
            header_para.clear()
            
            # Add image
            run = header_para.add_run()
            run.add_picture(image_path, width=Inches(1.5))
            
            return True
        except Exception as e:
            print(f"Could not add image {image_path}: {e}")
            return False

    # @staticmethod
    # def add_watermark_background(doc, bg_image_path="templates/bg.png"):
    #     """Add a watermark/background effect to the document matching the PDF template"""
    #     try:
    #         # Get the first section
    #         section = doc.sections[0]
            
    #         # Add watermark to header to make it appear on all pages
    #         header = section.header
            
    #         # Create a paragraph for the watermark
    #         watermark_para = header.add_paragraph()
    #         watermark_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
    #         # Add the background image as a watermark
    #         try:
    #             run = watermark_para.add_run()
    #             # Add picture with reduced size and transparency effect
    #             picture = run.add_picture(bg_image_path, width=Inches(3.0))
                
    #             # Access the drawing element to set transparency
    #             drawing = run._element.xpath('.//wp:anchor | .//wp:inline')[0]
                
    #             # Create transparency effect by modifying the picture properties
    #             # This creates a semi-transparent effect similar to CSS opacity: 0.3
    #             pic = drawing.xpath('.//pic:pic')[0]
    #             blipFill = pic.xpath('.//a:blipFill')[0]
                
    #             # Add alpha modulation for transparency (30% opacity = 70% transparency)
    #             alphaModFix = OxmlElement('a:alphaModFix')
    #             alphaModFix.set('amt', '30000')  # 30% opacity (30000 out of 100000)
                
    #             # Find or create the effect list
    #             effectLst = blipFill.find('.//a:effectLst')
    #             if effectLst is None:
    #                 effectLst = OxmlElement('a:effectLst')
    #                 blipFill.append(effectLst)
                
    #             effectLst.append(alphaModFix)
                
    #         except Exception as e:
    #             print(f"Could not add background image watermark: {e}")
    #             # Fallback: Add a light text watermark
    #             watermark_run = watermark_para.add_run("ShorthillsAI")
    #             watermark_run.font.size = Pt(72)
    #             watermark_run.font.color.rgb = RGBColor(245, 245, 245)  # Very light grey
            
    #         # Position the watermark behind text
    #         watermark_para.paragraph_format.space_before = Pt(0)
    #         watermark_para.paragraph_format.space_after = Pt(0)
            
    #         return True
    #     except Exception as e:
    #         print(f"Could not add watermark background: {e}")
    #         return False

    @staticmethod
    def add_logo_header(doc, left_logo_path="templates/left_logo_small.png", 
                       right_logo_path="templates/right_logo_small.png"):
        """Add logos to document header"""
        try:
            section = doc.sections[0]
            header = section.header
            
            # Clear existing header content
            header.paragraphs[0].clear()
            
            # Create a table for logo layout
            logo_table = header.add_table(rows=1, cols=3)
            logo_table.autofit = False
            
            # Set column widths
            logo_table.columns[0].width = Inches(2.0)  # Left logo
            logo_table.columns[1].width = Inches(3.5)  # Center space
            logo_table.columns[2].width = Inches(2.0)  # Right logo
            
            # Left logo
            left_cell = logo_table.cell(0, 0)
            left_para = left_cell.paragraphs[0]
            left_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            try:
                left_run = left_para.add_run()
                left_run.add_picture(left_logo_path, width=Inches(1.2))
            except:
                # Fallback text if image not found
                left_run = left_para.add_run("ShorthillsAI")
                left_run.font.size = Pt(10)
                left_run.font.color.rgb = RGBColor(242, 93, 93)
            
            # Right logo
            right_cell = logo_table.cell(0, 2)
            right_para = right_cell.paragraphs[0]
            right_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            try:
                right_run = right_para.add_run()
                right_run.add_picture(right_logo_path, width=Inches(1.2))
            except:
                # Fallback text if image not found
                right_run = right_para.add_run("Microsoft Partner")
                right_run.font.size = Pt(8)
                right_run.font.color.rgb = RGBColor(102, 102, 102)
            
            # Remove table borders
            for row in logo_table.rows:
                for cell in row.cells:
                    cell._tc.get_or_add_tcPr().append(
                        OxmlElement('w:tcBorders')
                    )
            
            return True
        except Exception as e:
            print(f"Could not add logo header: {e}")
            return False

    @staticmethod
    def add_grey_sidebar(table, left_cell):
        """Add grey background to left column to match template"""
        try:
            # Add shading to the left cell
            shading_elm = OxmlElement('w:shd')
            shading_elm.set(qn('w:fill'), 'EEEEEE')  # Light grey background
            left_cell._tc.get_or_add_tcPr().append(shading_elm)
            
            # Add some padding
            tc_pr = left_cell._tc.get_or_add_tcPr()
            tc_mar = OxmlElement('w:tcMar')
            
            # Set margins
            for margin in ['top', 'left', 'bottom', 'right']:
                mar_elem = OxmlElement(f'w:{margin}')
                mar_elem.set(qn('w:w'), '120')  # 120 twentieths of a point
                mar_elem.set(qn('w:type'), 'dxa')
                tc_mar.append(mar_elem)
            
            tc_pr.append(tc_mar)
            return True
        except Exception as e:
            print(f"Could not add grey sidebar: {e}")
            return False


    @staticmethod
    def generate_docx(data, keywords=None, left_logo_path="templates/left_logo_small.png", right_logo_path="templates/right_logo_small.png"):
        """
        Generate a .docx resume from the provided data, matching the HTML template layout as closely as possible.
        Supports keyword bolding and multi-page layout.
        Returns a BytesIO object containing the Word file.
        """
        def clean_html_text(text):
            if not text:
                return ""
            text = str(text)
            bold_parts = []
            import re
            strong_pattern = r'<strong>(.*?)</strong>'
            matches = list(re.finditer(strong_pattern, text, re.IGNORECASE))
            if matches:
                current_pos = 0
                for match in matches:
                    if match.start() > current_pos:
                        bold_parts.append((text[current_pos:match.start()], False))
                    bold_parts.append((match.group(1), True))
                    current_pos = match.end()
                if current_pos < len(text):
                    bold_parts.append((text[current_pos:], False))
            else:
                bold_parts = [(text, False)]
            cleaned_parts = []
            for part_text, is_bold in bold_parts:
                clean_text = re.sub(r'<[^>]+>', '', part_text)
                cleaned_parts.append((clean_text, is_bold))
            return cleaned_parts

        def add_formatted_text(paragraph, text_parts):
            for text, is_bold in text_parts:
                if text.strip():
                    run = paragraph.add_run(text)
                    run.font.name = 'Montserrat'
                    run.font.size = Pt(10)
                    if is_bold:
                        run.bold = True

        def add_section_title(cell, title, margin_top=18):
            para = cell.add_paragraph()
            para.paragraph_format.space_before = Pt(margin_top)
            para.paragraph_format.space_after = Pt(4)
            run = para.add_run(title.upper())
            run.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(242, 93, 93)
            run.font.name = 'Montserrat'
            return para

        def add_bullet_point(cell, text_parts, indent=10):
            para = cell.add_paragraph()
            para.paragraph_format.left_indent = Pt(indent)
            para.paragraph_format.space_after = Pt(2)
            bullet_run = para.add_run("➔ ")
            bullet_run.font.name = 'Montserrat'
            bullet_run.font.size = Pt(10)
            bullet_run.font.color.rgb = RGBColor(242, 93, 93)
            add_formatted_text(para, text_parts)
            return para

        data_copy = copy.deepcopy(data)
        doc = docx.Document()
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.5)
            section.bottom_margin = Inches(0.5)
            section.left_margin = Inches(0.5)
            section.right_margin = Inches(0.5)

        # --- HEADER WITH LOGOS ---
        header = doc.sections[0].header
        header_table = header.add_table(rows=1, cols=3, width=docx.shared.Inches(7.0))
        header_table.autofit = False
        header_table.columns[0].width = Inches(1.5)
        header_table.columns[1].width = Inches(4.0)
        header_table.columns[2].width = Inches(1.5)
        # Left logo - increased size to match PDF template
        left_cell = header_table.cell(0, 0)
        left_para = left_cell.paragraphs[0]
        left_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        try:
            left_para.add_run().add_picture(left_logo_path, width=Inches(1.5))
        except Exception:
            left_para.add_run("ShorthillsAI")
        # Center (empty)
        center_cell = header_table.cell(0, 1)
        # Right logo
        right_cell = header_table.cell(0, 2)
        right_para = right_cell.paragraphs[0]
        right_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        try:
            right_para.add_run().add_picture(right_logo_path, width=Inches(1.1))
        except Exception:
            right_para.add_run("Microsoft Partner")



        # --- ADD BORDER TO DOCUMENT ---
        # Add border to the page
        for section in doc.sections:
            sectPr = section._sectPr
            pgBorders = OxmlElement('w:pgBorders')
            pgBorders.set(qn('w:offsetFrom'), 'page')
            
            # Add border on all sides
            for border_name in ['top', 'left', 'bottom', 'right']:
                border = OxmlElement(f'w:{border_name}')
                border.set(qn('w:val'), 'single')
                border.set(qn('w:sz'), '18')  # 3pt border (18 eighths of a point)
                border.set(qn('w:space'), '24')
                border.set(qn('w:color'), 'F25D5D')  # Red color matching template
                pgBorders.append(border)
            
            sectPr.append(pgBorders)

        # --- MAIN CONTENT TABLE (2 columns) ---
        table = doc.add_table(rows=1, cols=2)
        table.autofit = False
        table.allow_autofit = False
        table.columns[0].width = Inches(2.8)
        table.columns[1].width = Inches(4.7)
        left_cell = table.cell(0, 0)
        right_cell = table.cell(0, 1)
        # Remove default paragraphs
        left_cell._tc.clear_content()
        right_cell._tc.clear_content()

        # --- GREY BOX FOR NAME/TITLE ---
        name_title_table = left_cell.add_table(rows=2, cols=1)
        name_title_table.autofit = False
        name_title_table.columns[0].width = Inches(2.7)
        # Add shading to the cell (grey background)
        name_cell = name_title_table.cell(0, 0)
        name_para = name_cell.paragraphs[0]
        name_para.paragraph_format.space_before = Pt(8)
        name_para.paragraph_format.space_after = Pt(0)
        name_run = name_para.add_run(data_copy.get('name', ''))
        name_run.bold = True
        name_run.font.size = Pt(20)
        name_run.font.color.rgb = RGBColor(242, 93, 93)
        name_run.font.name = 'Montserrat'
        # Add shading
        shading_elm = OxmlElement('w:shd')
        shading_elm.set(qn('w:fill'), 'F2F2F2')
        name_cell._tc.get_or_add_tcPr().append(shading_elm)
        # Title
        title_cell = name_title_table.cell(1, 0)
        title_para = title_cell.paragraphs[0]
        title_para.paragraph_format.space_after = Pt(10)
        title_parts = clean_html_text(data_copy.get('title', ''))
        for text, is_bold in title_parts:
            if text.strip():
                title_run = title_para.add_run(text)
                title_run.font.size = Pt(14)
                title_run.font.name = 'Montserrat'
                title_run.bold = True if is_bold else False
        # Add shading to title cell
        shading_elm2 = OxmlElement('w:shd')
        shading_elm2.set(qn('w:fill'), 'F2F2F2')
        title_cell._tc.get_or_add_tcPr().append(shading_elm2)
        # Add some space after the name/title box
        left_cell.add_paragraph().add_run("")

        # --- LEFT COLUMN CONTENT ---
        # Skills
        if data_copy.get('skills'):
            add_section_title(left_cell, 'Skills', margin_top=27)
            space_para = left_cell.add_paragraph()
            space_para.paragraph_format.space_after = Pt(12)
            for skill in data_copy['skills']:
                skill_parts = clean_html_text(skill)
                add_bullet_point(left_cell, skill_parts)
        # Education
        if data_copy.get('education'):
            add_section_title(left_cell, 'Education')
            for edu in data_copy['education']:
                if isinstance(edu, dict):
                    para = left_cell.add_paragraph()
                    para.paragraph_format.left_indent = Pt(10)
                    para.paragraph_format.space_after = Pt(4)
                    bullet_run = para.add_run("➔ ")
                    bullet_run.font.name = 'Montserrat'
                    bullet_run.font.size = Pt(10)
                    bullet_run.font.color.rgb = RGBColor(242, 93, 93)
                    if edu.get('degree'):
                        degree_parts = clean_html_text(edu['degree'])
                        add_formatted_text(para, degree_parts)
                    if edu.get('institution'):
                        para.add_run('\n')
                        inst_parts = clean_html_text(edu['institution'])
                        for text, is_bold in inst_parts:
                            if text.strip():
                                inst_run = para.add_run(text)
                                inst_run.font.name = 'Montserrat'
                                inst_run.font.size = Pt(10)
                                inst_run.bold = True
                else:
                    edu_parts = clean_html_text(str(edu))
                    add_bullet_point(left_cell, edu_parts)
        # Certifications
        if data_copy.get('certifications'):
            add_section_title(left_cell, 'Certifications')
            for cert in data_copy['certifications']:
                para = left_cell.add_paragraph()
                para.paragraph_format.left_indent = Pt(10)
                para.paragraph_format.space_after = Pt(4)
                bullet_run = para.add_run("➔ ")
                bullet_run.font.name = 'Montserrat'
                bullet_run.font.size = Pt(10)
                bullet_run.font.color.rgb = RGBColor(242, 93, 93)
                if isinstance(cert, dict):
                    if cert.get('title'):
                        title_parts = clean_html_text(cert['title'])
                        add_formatted_text(para, title_parts)
                    if cert.get('issuer'):
                        para.add_run('\n')
                        issuer_parts = clean_html_text(cert['issuer'])
                        for text, is_bold in issuer_parts:
                            if text.strip():
                                issuer_run = para.add_run(text)
                                issuer_run.font.name = 'Montserrat'
                                issuer_run.font.size = Pt(9)
                                issuer_run.font.color.rgb = RGBColor(102, 102, 102)
                                issuer_run.bold = True if is_bold else False
                    if cert.get('year'):
                        para.add_run(f"\n{cert['year']}")
                else:
                    cert_parts = clean_html_text(str(cert))
                    add_formatted_text(para, cert_parts)

        # --- RIGHT COLUMN CONTENT ---
        # Summary
        if data_copy.get('summary'):
            add_section_title(right_cell, 'Summary', margin_top=0)
            summary_para = right_cell.add_paragraph()
            summary_para.paragraph_format.space_after = Pt(18)
            summary_parts = clean_html_text(data_copy['summary'])
            add_formatted_text(summary_para, summary_parts)
        # Projects
        if data_copy.get('projects'):
            space_para = right_cell.add_paragraph()
            space_para.paragraph_format.space_after = Pt(4)
            add_section_title(right_cell, 'Key Responsibilities:', margin_top=19)
            for idx, project in enumerate(data_copy['projects']):
                if project.get('title'):
                    proj_title_para = right_cell.add_paragraph()
                    proj_title_para.paragraph_format.space_before = Pt(12)
                    proj_title_para.paragraph_format.space_after = Pt(4)
                    title_run = proj_title_para.add_run(f"Project {idx + 1}: ")
                    title_run.bold = True
                    title_run.font.size = Pt(11)
                    title_run.font.color.rgb = RGBColor(242, 93, 93)
                    title_run.font.name = 'Montserrat'
                    title_parts = clean_html_text(project['title'])
                    for text, is_bold in title_parts:
                        if text.strip():
                            proj_run = proj_title_para.add_run(text)
                            proj_run.bold = True
                            proj_run.font.size = Pt(11)
                            proj_run.font.color.rgb = RGBColor(242, 93, 93)
                            proj_run.font.name = 'Montserrat'
                if project.get('description'):
                    desc_text = project['description']
                    if isinstance(desc_text, str):
                        bullets = desc_text.split('\n')
                        bullets = [b.strip() for b in bullets if b.strip()]
                        for bullet in bullets:
                            bullet_para = right_cell.add_paragraph()
                            bullet_para.paragraph_format.left_indent = Pt(20)
                            bullet_para.paragraph_format.space_after = Pt(3)
                            bullet_run = bullet_para.add_run("➔ ")
                            bullet_run.font.name = 'Montserrat'
                            bullet_run.font.size = Pt(10)
                            bullet_run.font.color.rgb = RGBColor(242, 93, 93)
                            bullet_parts = clean_html_text(bullet)
                            add_formatted_text(bullet_para, bullet_parts)
        
        # --- ADD FOOTER ---
        footer = doc.sections[0].footer
        footer_para = footer.add_paragraph()
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer_para.paragraph_format.space_before = Pt(8)
        footer_para.paragraph_format.space_after = Pt(8)
        
        # Add red background to footer paragraph to match PDF template
        shading_elm = OxmlElement('w:shd')
        shading_elm.set(qn('w:fill'), 'F25D5D')  # Red background matching template
        footer_para._p.get_or_add_pPr().append(shading_elm)
        
        # Add footer text with white color to match PDF template
        footer_run = footer_para.add_run("© www.shorthills.ai")
        footer_run.font.name = 'Montserrat'
        footer_run.font.size = Pt(12)
        footer_run.font.color.rgb = RGBColor(255, 255, 255)  # White color for red background
        
        # Save document to BytesIO
        docx_file = io.BytesIO()
        doc.save(docx_file)
        docx_file.seek(0)
        return docx_file

    @staticmethod
    def generate_multi_page_docx(data, keywords=None):
        """
        Generate a multi-page .docx resume with proper page breaks and section distribution.
        This handles large amounts of content that don't fit on a single page.
        """
        
        def clean_html_text(text):
            """Remove HTML tags and convert to plain text with proper formatting"""
            if not text:
                return ""
            
            # Convert <strong> tags to track bold formatting
            text = str(text)
            bold_parts = []
            
            # Find all <strong> tags and their content
            import re
            strong_pattern = r'<strong>(.*?)</strong>'
            matches = list(re.finditer(strong_pattern, text, re.IGNORECASE))
            
            if matches:
                current_pos = 0
                for match in matches:
                    # Add text before bold part
                    if match.start() > current_pos:
                        bold_parts.append((text[current_pos:match.start()], False))
                    # Add bold part
                    bold_parts.append((match.group(1), True))
                    current_pos = match.end()
                # Add remaining text
                if current_pos < len(text):
                    bold_parts.append((text[current_pos:], False))
            else:
                bold_parts = [(text, False)]
            
            # Clean any remaining HTML tags
            cleaned_parts = []
            for part_text, is_bold in bold_parts:
                clean_text = re.sub(r'<[^>]+>', '', part_text)
                cleaned_parts.append((clean_text, is_bold))
            
            return cleaned_parts

        def add_formatted_text(paragraph, text_parts):
            """Add text with mixed formatting to a paragraph"""
            for text, is_bold in text_parts:
                if text.strip():
                    run = paragraph.add_run(text)
                    run.font.name = 'Montserrat'
                    run.font.size = Pt(10)
                    if is_bold:
                        run.bold = True

        def add_section_title(container, title, margin_top=18):
            """Add a section title with consistent formatting"""
            para = container.add_paragraph()
            para.paragraph_format.space_before = Pt(margin_top)
            para.paragraph_format.space_after = Pt(4)
            run = para.add_run(title.upper())
            run.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(242, 93, 93)  # #f25d5d
            run.font.name = 'Montserrat'
            return para

        def estimate_content_size(data):
            """Estimate how much content we have to determine if multi-page is needed"""
            size = 0
            size += len(data.get('skills', [])) * 2  # Each skill takes ~2 units
            size += len(data.get('education', [])) * 4  # Each education entry ~4 units
            size += len(data.get('certifications', [])) * 3  # Each cert ~3 units
            
            # Projects can be large
            for project in data.get('projects', []):
                project_size = 5  # Base size for title
                if project.get('description'):
                    bullets = project['description'].split('\n') if isinstance(project['description'], str) else []
                    project_size += len([b for b in bullets if b.strip()]) * 2
                size += project_size
            
            return size

        # Apply keyword bolding if keywords provided (using same logic as PDF version)
        data_copy = copy.deepcopy(data) if keywords else data
        
        doc = docx.Document()
        
        # Set page margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.5)
            section.bottom_margin = Inches(0.5)
            section.left_margin = Inches(0.5)
            section.right_margin = Inches(0.5)

        # --- ADD HEADER WITH LOGOS ---
        header = doc.sections[0].header
        header_table = header.add_table(rows=1, cols=3, width=docx.shared.Inches(7.0))
        header_table.autofit = False
        header_table.columns[0].width = Inches(1.5)
        header_table.columns[1].width = Inches(4.0)
        header_table.columns[2].width = Inches(1.5)
        # Left logo - increased size to match PDF template
        left_cell = header_table.cell(0, 0)
        left_para = left_cell.paragraphs[0]
        left_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        try:
            left_para.add_run().add_picture("templates/left_logo_small.png", width=Inches(1.5))
        except Exception:
            left_para.add_run("ShorthillsAI")
        # Center (empty)
        center_cell = header_table.cell(0, 1)
        # Right logo
        right_cell = header_table.cell(0, 2)
        right_para = right_cell.paragraphs[0]
        right_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        try:
            right_para.add_run().add_picture("templates/right_logo_small.png", width=Inches(1.1))
        except Exception:
            right_para.add_run("Microsoft Partner")



        # --- ADD BORDER TO DOCUMENT ---
        # Add border to the page
        for section in doc.sections:
            sectPr = section._sectPr
            pgBorders = OxmlElement('w:pgBorders')
            pgBorders.set(qn('w:offsetFrom'), 'page')
            
            # Add border on all sides
            for border_name in ['top', 'left', 'bottom', 'right']:
                border = OxmlElement(f'w:{border_name}')
                border.set(qn('w:val'), 'single')
                border.set(qn('w:sz'), '18')  # 3pt border (18 eighths of a point)
                border.set(qn('w:space'), '24')
                border.set(qn('w:color'), 'F25D5D')  # Red color matching template
                pgBorders.append(border)
            
            sectPr.append(pgBorders)

        # Estimate if we need multiple pages
        content_size = estimate_content_size(data_copy)
        needs_multiple_pages = content_size > 60  # Threshold for single page

        if not needs_multiple_pages:
            # Use single page layout
            return PDFUtils.generate_docx(data_copy, keywords)

        # Multi-page layout
        # Page 1: Header, Name, Summary, First few projects
        
        # Add header
        header_para = doc.add_paragraph()
        header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        header_run = header_para.add_run("COMPANY LOGOS")
        header_run.font.size = Pt(8)
        header_run.font.color.rgb = RGBColor(128, 128, 128)
        header_para.paragraph_format.space_after = Pt(24)

        # Name and title
        name_para = doc.add_paragraph()
        name_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        name_para.paragraph_format.space_after = Pt(6)
        name_run = name_para.add_run(data_copy.get('name', ''))
        name_run.bold = True
        name_run.font.size = Pt(24)
        name_run.font.color.rgb = RGBColor(242, 93, 93)
        name_run.font.name = 'Montserrat'
        
        if data_copy.get('title'):
            title_para = doc.add_paragraph()
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_para.paragraph_format.space_after = Pt(20)
            title_parts = clean_html_text(data_copy['title'])
            for text, is_bold in title_parts:
                if text.strip():
                    title_run = title_para.add_run(text)
                    title_run.font.size = Pt(16)
                    title_run.font.name = 'Montserrat'
                    title_run.bold = True if is_bold else False

        # Summary section
        if data_copy.get('summary'):
            add_section_title(doc, 'Summary')
            summary_para = doc.add_paragraph()
            summary_para.paragraph_format.space_after = Pt(18)
            summary_parts = clean_html_text(data_copy['summary'])
            add_formatted_text(summary_para, summary_parts)

        # First page projects (limit to 2-3 projects)
        projects = data_copy.get('projects', [])
        first_page_projects = projects[:3]  # First 3 projects on page 1
        remaining_projects = projects[3:]
        
        if first_page_projects:
            add_section_title(doc, 'Key Responsibilities')
            
            for idx, project in enumerate(first_page_projects):
                # Project title
                if project.get('title'):
                    proj_title_para = doc.add_paragraph()
                    proj_title_para.paragraph_format.space_before = Pt(12)
                    proj_title_para.paragraph_format.space_after = Pt(4)
                    
                    title_run = proj_title_para.add_run(f"{idx + 1}. ")
                    title_run.bold = True
                    title_run.font.size = Pt(11)
                    title_run.font.color.rgb = RGBColor(242, 93, 93)
                    title_run.font.name = 'Montserrat'
                    
                    title_parts = clean_html_text(project['title'])
                    for text, is_bold in title_parts:
                        if text.strip():
                            proj_run = proj_title_para.add_run(text)
                            proj_run.bold = True
                            proj_run.font.size = Pt(11)
                            proj_run.font.color.rgb = RGBColor(242, 93, 93)
                            proj_run.font.name = 'Montserrat'

                # Project description
                if project.get('description'):
                    desc_text = project['description']
                    if isinstance(desc_text, str):
                        bullets = desc_text.split('\n')
                        bullets = [b.strip() for b in bullets if b.strip()]
                        
                        for bullet in bullets[:4]:  # Limit bullets per project on first page
                            bullet_para = doc.add_paragraph()
                            bullet_para.paragraph_format.left_indent = Pt(20)
                            bullet_para.paragraph_format.space_after = Pt(3)
                            
                            bullet_run = bullet_para.add_run("• ")
                            bullet_run.font.name = 'Montserrat'
                            bullet_run.font.size = Pt(10)
                            bullet_run.font.color.rgb = RGBColor(242, 93, 93)
                            
                            bullet_parts = clean_html_text(bullet)
                            add_formatted_text(bullet_para, bullet_parts)

        # Page break for second page
        if remaining_projects or data_copy.get('skills') or data_copy.get('education') or data_copy.get('certifications'):
            doc.add_page_break()

        # Page 2+: Remaining projects and left column content
        if remaining_projects:
            add_section_title(doc, 'Additional Responsibilities', margin_top=0)
            
            for idx, project in enumerate(remaining_projects):
                proj_idx = idx + len(first_page_projects) + 1
                
                if project.get('title'):
                    proj_title_para = doc.add_paragraph()
                    proj_title_para.paragraph_format.space_before = Pt(12)
                    proj_title_para.paragraph_format.space_after = Pt(4)
                    
                    title_run = proj_title_para.add_run(f"{proj_idx}. ")
                    title_run.bold = True
                    title_run.font.size = Pt(11)
                    title_run.font.color.rgb = RGBColor(242, 93, 93)
                    title_run.font.name = 'Montserrat'
                    
                    title_parts = clean_html_text(project['title'])
                    for text, is_bold in title_parts:
                        if text.strip():
                            proj_run = proj_title_para.add_run(text)
                            proj_run.bold = True
                            proj_run.font.size = Pt(11)
                            proj_run.font.color.rgb = RGBColor(242, 93, 93)
                            proj_run.font.name = 'Montserrat'

                if project.get('description'):
                    desc_text = project['description']
                    if isinstance(desc_text, str):
                        bullets = desc_text.split('\n')
                        bullets = [b.strip() for b in bullets if b.strip()]
                        
                        for bullet in bullets:
                            bullet_para = doc.add_paragraph()
                            bullet_para.paragraph_format.left_indent = Pt(20)
                            bullet_para.paragraph_format.space_after = Pt(3)
                            
                            bullet_run = bullet_para.add_run("• ")
                            bullet_run.font.name = 'Montserrat'
                            bullet_run.font.size = Pt(10)
                            bullet_run.font.color.rgb = RGBColor(242, 93, 93)
                            
                            bullet_parts = clean_html_text(bullet)
                            add_formatted_text(bullet_para, bullet_parts)

        # Skills section
        if data_copy.get('skills'):
            add_section_title(doc, 'Skills')
            
            for skill in data_copy['skills']:
                skill_para = doc.add_paragraph()
                skill_para.paragraph_format.left_indent = Pt(20)
                skill_para.paragraph_format.space_after = Pt(3)
                
                bullet_run = skill_para.add_run("• ")
                bullet_run.font.name = 'Montserrat'
                bullet_run.font.size = Pt(10)
                bullet_run.font.color.rgb = RGBColor(242, 93, 93)
                
                skill_parts = clean_html_text(skill)
                add_formatted_text(skill_para, skill_parts)

        # Education section
        if data_copy.get('education'):
            add_section_title(doc, 'Education')
            
            for edu in data_copy['education']:
                edu_para = doc.add_paragraph()
                edu_para.paragraph_format.left_indent = Pt(20)
                edu_para.paragraph_format.space_after = Pt(6)
                
                bullet_run = edu_para.add_run("• ")
                bullet_run.font.name = 'Montserrat'
                bullet_run.font.size = Pt(10)
                bullet_run.font.color.rgb = RGBColor(242, 93, 93)
                
                if isinstance(edu, dict):
                    if edu.get('degree'):
                        degree_parts = clean_html_text(edu['degree'])
                        add_formatted_text(edu_para, degree_parts)
                    
                    if edu.get('institution'):
                        edu_para.add_run('\n      ')  # Indent for institution
                        inst_parts = clean_html_text(edu['institution'])
                        for text, is_bold in inst_parts:
                            if text.strip():
                                inst_run = edu_para.add_run(text)
                                inst_run.font.name = 'Montserrat'
                                inst_run.font.size = Pt(10)
                                inst_run.bold = True
                else:
                    edu_parts = clean_html_text(str(edu))
                    add_formatted_text(edu_para, edu_parts)

        # Certifications section
        if data_copy.get('certifications'):
            add_section_title(doc, 'Certifications')
            
            for cert in data_copy['certifications']:
                cert_para = doc.add_paragraph()
                cert_para.paragraph_format.left_indent = Pt(20)
                cert_para.paragraph_format.space_after = Pt(6)
                
                bullet_run = cert_para.add_run("• ")
                bullet_run.font.name = 'Montserrat'
                bullet_run.font.size = Pt(10)
                bullet_run.font.color.rgb = RGBColor(242, 93, 93)
                
                if isinstance(cert, dict):
                    if cert.get('title'):
                        title_parts = clean_html_text(cert['title'])
                        add_formatted_text(cert_para, title_parts)
                    
                    if cert.get('issuer'):
                        cert_para.add_run('\n      ')  # Indent for issuer
                        issuer_parts = clean_html_text(cert['issuer'])
                        for text, is_bold in issuer_parts:
                            if text.strip():
                                issuer_run = cert_para.add_run(text)
                                issuer_run.font.name = 'Montserrat'
                                issuer_run.font.size = Pt(9)
                                issuer_run.font.color.rgb = RGBColor(102, 102, 102)
                                issuer_run.bold = True if is_bold else False
                    
                    if cert.get('year'):
                        cert_para.add_run(f"\n      {cert['year']}")
                else:
                    cert_parts = clean_html_text(str(cert))
                    add_formatted_text(cert_para, cert_parts)

        # --- ADD FOOTER ---
        footer = doc.sections[0].footer
        footer_para = footer.add_paragraph()
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer_para.paragraph_format.space_before = Pt(8)
        footer_para.paragraph_format.space_after = Pt(8)
        
        # Add red background to footer paragraph to match PDF template
        shading_elm = OxmlElement('w:shd')
        shading_elm.set(qn('w:fill'), 'F25D5D')  # Red background matching template
        footer_para._p.get_or_add_pPr().append(shading_elm)
        
        # Add footer text with white color to match PDF template
        footer_run = footer_para.add_run("© www.shorthills.ai")
        footer_run.font.name = 'Montserrat'
        footer_run.font.size = Pt(12)
        footer_run.font.color.rgb = RGBColor(255, 255, 255)  # White color for red background

        # Save document
        docx_file = io.BytesIO()
        doc.save(docx_file)
        docx_file.seek(0)
        return docx_file