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
    def clean_html_text(text):
        """Clean HTML tags and return text with bold formatting info"""
        if not text:
            return []
        
        text = str(text)
        bold_parts = []
        
        # Find all <strong> tags and their content
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
            if clean_text.strip():
                cleaned_parts.append((clean_text, is_bold))
        
        return cleaned_parts

    @staticmethod
    def add_formatted_text(paragraph, text_parts, font_size=10, font_color=RGBColor(34, 34, 34)):
        """Add text with mixed formatting to a paragraph"""
        for text, is_bold in text_parts:
            if text.strip():
                run = paragraph.add_run(text)
                run.font.name = 'Montserrat'
                run.font.size = Pt(font_size)
                run.font.color.rgb = font_color
                if is_bold:
                    run.bold = True

    @staticmethod
    def add_section_title(container, title, margin_top=18):
        """Add a section title with consistent formatting matching PDF template"""
        para = container.add_paragraph()
        para.paragraph_format.space_before = Pt(margin_top)
        para.paragraph_format.space_after = Pt(4)
        run = para.add_run(title.upper())
        run.bold = True
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(242, 93, 93)  # #f25d5d
        run.font.name = 'Montserrat'
        return para

    @staticmethod
    def add_triangle_bullet_point(container, text_parts, indent=22):
        """Add bullet point with triangle (▶) matching PDF template"""
        para = container.add_paragraph()
        para.paragraph_format.left_indent = Pt(indent)
        para.paragraph_format.space_after = Pt(6)
        
        # Add red triangle
        triangle_run = para.add_run("▶ ")
        triangle_run.font.name = 'Montserrat'
        triangle_run.font.size = Pt(13)
        triangle_run.font.color.rgb = RGBColor(242, 93, 93)
        triangle_run.bold = True
        
        # Add text content
        DocxUtils.add_formatted_text(para, text_parts, font_size=12)
        return para

    @staticmethod
    def add_page_border(doc):
        """Add orange border optimized for Microsoft Word"""
        for section in doc.sections:
            sectPr = section._sectPr
            pgBorders = OxmlElement('w:pgBorders')
            pgBorders.set(qn('w:offsetFrom'), 'page')
            pgBorders.set(qn('w:display'), 'allPages')  # Word-specific: show on all pages
            
            # Add border on all sides - optimized for Word rendering
            for border_name in ['top', 'left', 'bottom', 'right']:
                border = OxmlElement(f'w:{border_name}')
                border.set(qn('w:val'), 'single')
                border.set(qn('w:sz'), '4')  # 0.5pt border (4/8ths of a point)
                border.set(qn('w:space'), '24')  # Use '24' for standard page distance
                border.set(qn('w:color'), 'F25D5D')  # Match resume's red color
                border.set(qn('w:themeColor'), 'none')  # Don't use theme colors
                pgBorders.append(border)
            
            sectPr.append(pgBorders)

    @staticmethod
    def optimize_table_for_word(table):
        """Optimize table properties specifically for Microsoft Word"""
        tbl = table._tbl
        tblPr = tbl.tblPr
        
        # Set table layout to fixed for consistent rendering in Word
        tblLayout = OxmlElement('w:tblLayout')
        tblLayout.set(qn('w:type'), 'fixed')
        tblPr.append(tblLayout)
        
        # Set table positioning
        tblpPr = OxmlElement('w:tblpPr')
        tblpPr.set(qn('w:leftFromText'), '0')
        tblpPr.set(qn('w:rightFromText'), '0')
        tblpPr.set(qn('w:vertAnchor'), 'page')
        tblpPr.set(qn('w:horzAnchor'), 'page')
        tblPr.append(tblpPr)
        
        # Remove table borders for seamless layout
        DocxUtils.remove_table_borders(table)

    @staticmethod
    def add_word_optimized_spacing(paragraph, space_before=0, space_after=0, line_spacing=1.0):
        """Add Word-optimized paragraph spacing"""
        pPr = paragraph._p.get_or_add_pPr()
        
        # Set spacing before
        if space_before > 0:
            spacingBefore = OxmlElement('w:spacing')
            spacingBefore.set(qn('w:before'), str(int(space_before * 20)))  # Convert pt to twentieths
            pPr.append(spacingBefore)
        
        # Set spacing after
        if space_after > 0:
            spacingAfter = OxmlElement('w:spacing')
            spacingAfter.set(qn('w:after'), str(int(space_after * 20)))  # Convert pt to twentieths
            pPr.append(spacingAfter)
        
        # Set line spacing for better readability in Word
        spacing = OxmlElement('w:spacing')
        spacing.set(qn('w:line'), str(int(line_spacing * 240)))  # 240 = single spacing in Word
        spacing.set(qn('w:lineRule'), 'auto')
        pPr.append(spacing)

    @staticmethod
    def add_word_font_optimization(run, font_name='Montserrat', font_size=10, is_bold=False, color_rgb=None):
        """Optimize font rendering for Microsoft Word"""
        # Set font with Word-specific properties
        run.font.name = font_name
        run.font.size = Pt(font_size)
        
        # Add font fallbacks for better Word compatibility
        rPr = run._r.get_or_add_rPr()
        
        # Set font family with fallbacks
        rFonts = OxmlElement('w:rFonts')
        rFonts.set(qn('w:ascii'), font_name)
        rFonts.set(qn('w:hAnsi'), font_name)
        rFonts.set(qn('w:cs'), font_name)
        rFonts.set(qn('w:eastAsia'), font_name)
        rPr.append(rFonts)
        
        # Set bold if needed
        if is_bold:
            run.bold = True
            # Add explicit bold for Word
            bold = OxmlElement('w:b')
            rPr.append(bold)
        
        # Set color if provided
        if color_rgb:
            run.font.color.rgb = color_rgb

    @staticmethod
    def add_background_watermark(doc, bg_image_path="templates/bg.png"):
        """Add background watermark matching PDF template opacity"""
        try:
            section = doc.sections[0]
            header = section.header
            
            # Create a paragraph for the watermark
            watermark_para = header.add_paragraph()
            watermark_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            try:
                run = watermark_para.add_run()
                # Add picture with size matching template (35% of page)
                picture = run.add_picture(bg_image_path, width=Inches(2.8))
                
                # Access the drawing element to set transparency (17% opacity like template)
                drawing_elements = run._element.xpath('.//wp:anchor | .//wp:inline')
                if drawing_elements:
                    drawing = drawing_elements[0]
                    pic_elements = drawing.xpath('.//pic:pic')
                    if pic_elements:
                        pic = pic_elements[0]
                        blip_elements = pic.xpath('.//a:blipFill')
                        if blip_elements:
                            blipFill = blip_elements[0]
                            
                            # Add alpha modulation for transparency (17% opacity)
                            alphaModFix = OxmlElement('a:alphaModFix')
                            alphaModFix.set('amt', '17000')  # 17% opacity (17000 out of 100000)
                            
                            # Find or create the effect list
                            effectLst = blipFill.find('.//a:effectLst')
                            if effectLst is None:
                                effectLst = OxmlElement('a:effectLst')
                                blipFill.append(effectLst)
                            
                            effectLst.append(alphaModFix)
                
            except Exception as e:
                print(f"Could not add background image watermark: {e}")
            
            # Position the watermark behind text
            watermark_para.paragraph_format.space_before = Pt(0)
            watermark_para.paragraph_format.space_after = Pt(0)
            
            return True
        except Exception as e:
            print(f"Could not add watermark background: {e}")
            return False

    @staticmethod
    def add_grey_background(cell):
        """Add grey background to cell matching PDF template"""
        try:
            shading_elm = OxmlElement('w:shd')
            shading_elm.set(qn('w:fill'), 'F2F2F2')  # #f2f2f2 matching template
            cell._tc.get_or_add_tcPr().append(shading_elm)
            
            # Add padding
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_mar = OxmlElement('w:tcMar')
            
            # Set margins
            for margin in ['top', 'left', 'bottom', 'right']:
                mar_elem = OxmlElement(f'w:{margin}')
                mar_elem.set(qn('w:w'), '120')  # 120 twentieths of a point (about 8pt)
                mar_elem.set(qn('w:type'), 'dxa')
                tc_mar.append(mar_elem)
            
            tc_pr.append(tc_mar)
            return True
        except Exception as e:
            print(f"Could not add grey background: {e}")
            return False

    @staticmethod
    def remove_table_borders(table):
        """Remove all table borders to create seamless layout"""
        tbl = table._tbl
        tblPr = tbl.tblPr
        
        # Remove table borders
        tblBorders = OxmlElement('w:tblBorders')
        for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            border = OxmlElement(f'w:{border_name}')
            border.set(qn('w:val'), 'nil')
            tblBorders.append(border)
        tblPr.append(tblBorders)
        
        # Remove cell borders
        for row in table.rows:
            for cell in row.cells:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()
                tcBorders = OxmlElement('w:tcBorders')
                for border_name in ['top', 'left', 'bottom', 'right']:
                    border = OxmlElement(f'w:{border_name}')
                    border.set(qn('w:val'), 'nil')
                    tcBorders.append(border)
                tcPr.append(tcBorders)

    @staticmethod
    def generate_docx(data, keywords=None, left_logo_path="templates/left_logo_small.png", right_logo_path="templates/right_logo_small.png"):
        """
        Generate a .docx resume matching the PDF template exactly.
        Returns a BytesIO object containing the Word file.
        """

        data_copy = copy.deepcopy(data)
        doc = docx.Document()
        
        # Set page margins to ZERO to make border go to page edge
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0)
            section.bottom_margin = Inches(0)
            section.left_margin = Inches(0)
            section.right_margin = Inches(0)
            
            # Set header and footer distances to be minimal but visible
            section.header_distance = Inches(0.1)
            section.footer_distance = Inches(0.1)
            
            # Word-specific page setup
            section.page_width = Inches(8.5)   # Standard letter width
            section.page_height = Inches(11)   # Standard letter height

        # Add page border that goes to the edge - optimized for Word
        DocxUtils.add_page_border(doc)
        
        # Skip watermark for now as requested
        # DocxUtils.add_background_watermark(doc)

        # --- HEADER WITH LOGOS (Word-optimized) ---
        header = doc.sections[0].header
        header.is_linked_to_previous = False
        
        # Clear any default header content
        for para in header.paragraphs:
            p = para._element
            p.getparent().remove(p)
        
        # Create header table with Word-specific optimization
        header_table = header.add_table(rows=1, cols=3, width=docx.shared.Inches(8.2))
        header_table.autofit = False
        DocxUtils.optimize_table_for_word(header_table)
        
        # Set table to full width
        header_table.columns[0].width = Inches(2.8)
        header_table.columns[1].width = Inches(2.6)
        header_table.columns[2].width = Inches(2.8)
        
        # Left logo - optimized for Word
        left_cell = header_table.cell(0, 0)
        left_para = left_cell.paragraphs[0]
        left_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        DocxUtils.add_word_optimized_spacing(left_para, space_before=2, space_after=0)
        left_para.paragraph_format.left_indent = Pt(12)
        try:
            left_run = left_para.add_run()
            left_run.add_picture(left_logo_path, height=Inches(0.35))
        except Exception:
            left_run = left_para.add_run("ShorthillsAI")
            DocxUtils.add_word_font_optimization(left_run, 'Montserrat', 10, True, RGBColor(242, 93, 93))
        
        # Right logo - optimized for Word
        right_cell = header_table.cell(0, 2)
        right_para = right_cell.paragraphs[0]
        right_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        DocxUtils.add_word_optimized_spacing(right_para, space_before=2, space_after=0)
        right_para.paragraph_format.right_indent = Pt(12)
        try:
            right_run = right_para.add_run()
            right_run.add_picture(right_logo_path, height=Inches(0.45))
        except Exception:
            right_run = right_para.add_run("Microsoft Partner")
            right_run.font.name = 'Montserrat'
            right_run.font.size = Pt(10)
            right_run.font.color.rgb = RGBColor(102, 102, 102)
            
        # Remove spacing paragraph after header
        # spacing_para = doc.add_paragraph()
        # DocxUtils.add_word_optimized_spacing(spacing_para, space_after=1)

        # --- MAIN CONTENT TABLE (Word-optimized) ---
        main_table = doc.add_table(rows=1, cols=2)
        main_table.autofit = False
        main_table.allow_autofit = False
        DocxUtils.optimize_table_for_word(main_table)
        
        # Add a vertical line separator between columns
        left_cell_for_border = main_table.cell(0, 0)
        tcPr = left_cell_for_border._tc.get_or_add_tcPr()
        tcBorders = OxmlElement('w:tcBorders')
        right_border = OxmlElement('w:right')
        right_border.set(qn('w:val'), 'single')
        right_border.set(qn('w:sz'), '4') # 0.5pt
        right_border.set(qn('w:space'), '0')
        right_border.set(qn('w:color'), 'D3D3D3') # Light grey
        tcBorders.append(right_border)
        tcPr.append(tcBorders)
        
        # Adjust column widths for more space on the left
        main_table.columns[0].width = Inches(2.9)  # Left column (wider)
        main_table.columns[1].width = Inches(5.3)  # Right column
        
        left_cell = main_table.cell(0, 0)
        right_cell = main_table.cell(0, 1)
        
        # Clear default paragraphs
        left_cell._tc.clear_content()
        right_cell._tc.clear_content()

        # Add smaller padding to left cell
        left_cell_padding = left_cell.add_paragraph()
        left_cell_padding.paragraph_format.left_indent = Pt(12)

        # --- LEFT COLUMN (Word-optimized) ---
        # Name and Title container with grey background
        name_title_table = left_cell.add_table(rows=2, cols=1)
        name_title_table.autofit = False
        name_title_table.columns[0].width = Inches(2.8) # Match column width
        DocxUtils.optimize_table_for_word(name_title_table)
        
        # Name cell - Word optimized
        name_cell = name_title_table.cell(0, 0)
        DocxUtils.add_grey_background(name_cell)
        name_para = name_cell.paragraphs[0]
        DocxUtils.add_word_optimized_spacing(name_para, space_before=0, space_after=2)
        name_para.paragraph_format.left_indent = Pt(8)
        name_run = name_para.add_run(data_copy.get('name', ''))
        DocxUtils.add_word_font_optimization(name_run, 'Montserrat', 17, True, RGBColor(242, 93, 93))
        
        # Title cell - Word optimized
        title_cell = name_title_table.cell(1, 0)
        DocxUtils.add_grey_background(title_cell)
        title_para = title_cell.paragraphs[0]
        DocxUtils.add_word_optimized_spacing(title_para, space_after=10)
        title_para.paragraph_format.left_indent = Pt(8)
        title_parts = DocxUtils.clean_html_text(data_copy.get('title', ''))
        for text, is_bold in title_parts:
            if text.strip():
                title_run = title_para.add_run(text)
                DocxUtils.add_word_font_optimization(title_run, 'Montserrat', 13, True, RGBColor(34, 34, 34))

        # Skills Section - Word optimized
        if data_copy.get('skills'):
            skills_para = left_cell.add_paragraph()
            DocxUtils.add_word_optimized_spacing(skills_para, space_before=10, space_after=2)
            skills_para.paragraph_format.left_indent = Pt(12)
            skills_run = skills_para.add_run('SKILLS')
            DocxUtils.add_word_font_optimization(skills_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
            
            for skill in data_copy['skills']:
                skill_parts = DocxUtils.clean_html_text(skill)
                skill_para = left_cell.add_paragraph()
                DocxUtils.add_word_optimized_spacing(skill_para, space_after=3)
                skill_para.paragraph_format.left_indent = Pt(28)
                
                # Add red triangle with Word optimization
                arrow_run = skill_para.add_run("▶ ")
                DocxUtils.add_word_font_optimization(arrow_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
                
                # Add skill text with Word optimization
                for text, is_bold in skill_parts:
                    if text.strip():
                        skill_run = skill_para.add_run(text)
                        DocxUtils.add_word_font_optimization(skill_run, 'Montserrat', 10, is_bold, RGBColor(34, 34, 34))

        # Education Section - Word optimized
        if data_copy.get('education'):
            edu_para = left_cell.add_paragraph()
            DocxUtils.add_word_optimized_spacing(edu_para, space_before=10, space_after=2)
            edu_para.paragraph_format.left_indent = Pt(12)
            edu_run = edu_para.add_run('EDUCATION')
            DocxUtils.add_word_font_optimization(edu_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
            
            for edu in data_copy['education']:
                if isinstance(edu, dict):
                    para = left_cell.add_paragraph()
                    DocxUtils.add_word_optimized_spacing(para, space_after=3)
                    para.paragraph_format.left_indent = Pt(28)
                    
                    # Add red triangle
                    arrow_run = para.add_run("▶ ")
                    DocxUtils.add_word_font_optimization(arrow_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
                    
                    # Add degree
                    if edu.get('degree'):
                        degree_parts = DocxUtils.clean_html_text(edu['degree'])
                        for text, is_bold in degree_parts:
                            if text.strip():
                                degree_run = para.add_run(text)
                                DocxUtils.add_word_font_optimization(degree_run, 'Montserrat', 10, is_bold, RGBColor(34, 34, 34))
                    
                    # Add institution on new line
                    if edu.get('institution'):
                        para.add_run('\n')
                        inst_parts = DocxUtils.clean_html_text(edu['institution'])
                        for text, is_bold in inst_parts:
                            if text.strip():
                                inst_run = para.add_run(text)
                                DocxUtils.add_word_font_optimization(inst_run, 'Montserrat', 10, True, RGBColor(34, 34, 34))

        # Certifications Section - Word optimized
        if data_copy.get('certifications'):
            cert_para = left_cell.add_paragraph()
            DocxUtils.add_word_optimized_spacing(cert_para, space_before=10, space_after=2)
            cert_para.paragraph_format.left_indent = Pt(12)
            cert_run = cert_para.add_run('CERTIFICATIONS')
            DocxUtils.add_word_font_optimization(cert_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
            
            for cert in data_copy['certifications']:
                para = left_cell.add_paragraph()
                DocxUtils.add_word_optimized_spacing(para, space_after=3)
                para.paragraph_format.left_indent = Pt(28)
                
                # Add red triangle
                arrow_run = para.add_run("▶ ")
                DocxUtils.add_word_font_optimization(arrow_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
                
                if isinstance(cert, dict):
                    # Add certification title
                    if cert.get('title'):
                        title_parts = DocxUtils.clean_html_text(cert['title'])
                        for text, is_bold in title_parts:
                            if text.strip():
                                title_run = para.add_run(text)
                                DocxUtils.add_word_font_optimization(title_run, 'Montserrat', 10, is_bold, RGBColor(34, 34, 34))
                    
                    # Add issuer
                    if cert.get('issuer'):
                        para.add_run('\n')
                        issuer_parts = DocxUtils.clean_html_text(cert['issuer'])
                        for text, is_bold in issuer_parts:
                            if text.strip():
                                issuer_run = para.add_run(text)
                                DocxUtils.add_word_font_optimization(issuer_run, 'Montserrat', 10, True, RGBColor(34, 34, 34))
                    
                    # Add year
                    if cert.get('year'):
                        year_run = para.add_run(f"\n{cert['year']}")
                        DocxUtils.add_word_font_optimization(year_run, 'Montserrat', 10, False, RGBColor(34, 34, 34))

        # --- RIGHT COLUMN (Word-optimized) ---
        # Add right column padding
        right_cell_padding = right_cell.add_paragraph()
        right_cell_padding.paragraph_format.left_indent = Pt(12)

        # Summary Section - Word optimized
        if data_copy.get('summary'):
            summary_title_para = right_cell.add_paragraph()
            DocxUtils.add_word_optimized_spacing(summary_title_para, space_before=0, space_after=2)
            summary_title_para.paragraph_format.left_indent = Pt(12)
            summary_title_run = summary_title_para.add_run('SUMMARY')
            DocxUtils.add_word_font_optimization(summary_title_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
            
            summary_para = right_cell.add_paragraph()
            DocxUtils.add_word_optimized_spacing(summary_para, space_after=5)
            summary_para.paragraph_format.left_indent = Pt(12)
            summary_parts = DocxUtils.clean_html_text(data_copy['summary'])
            for text, is_bold in summary_parts:
                if text.strip():
                    summary_run = summary_para.add_run(text)
                    DocxUtils.add_word_font_optimization(summary_run, 'Montserrat', 10, is_bold, RGBColor(34, 34, 34))

        # Projects Section - Word optimized
        if data_copy.get('projects'):
            # Add minimal spacing
            spacing_para = right_cell.add_paragraph()
            DocxUtils.add_word_optimized_spacing(spacing_para, space_after=3)
            
            # Section title
            section_para = right_cell.add_paragraph()
            DocxUtils.add_word_optimized_spacing(section_para, space_after=2)
            section_para.paragraph_format.left_indent = Pt(12)
            section_run = section_para.add_run('KEY RESPONSIBILITIES:')
            DocxUtils.add_word_font_optimization(section_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
            
            # Add minimal spacing
            spacing_para2 = right_cell.add_paragraph()
            DocxUtils.add_word_optimized_spacing(spacing_para2, space_after=1)
            
            # Project entries
            for idx, project in enumerate(data_copy['projects']):
                # Project title
                if project.get('title'):
                    proj_title_para = right_cell.add_paragraph()
                    DocxUtils.add_word_optimized_spacing(proj_title_para, space_before=6, space_after=2)
                    proj_title_para.paragraph_format.left_indent = Pt(12)
                    
                    title_run = proj_title_para.add_run(f"Project {idx + 1}: ")
                    DocxUtils.add_word_font_optimization(title_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
                    
                    title_parts = DocxUtils.clean_html_text(project['title'])
                    for text, is_bold in title_parts:
                        if text.strip():
                            proj_run = proj_title_para.add_run(text)
                            DocxUtils.add_word_font_optimization(proj_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
                
                # Project description bullets
                if project.get('description'):
                    desc_text = project['description']
                    if isinstance(desc_text, str):
                        # Enhanced bullet splitting to handle HTML lists, newlines, and bullet characters
                        desc_text = desc_text.replace('</li>', '</li>\n').replace('•', '\n')
                        # Split by full stops, newlines, <br>, or <li>.
                        bullets = re.split(r'(?<=[.?!])\s+|\n|<br\s*/?>|<li>', desc_text)
                        
                        for bullet_html in bullets:
                            bullet_html = bullet_html.strip()
                            if not bullet_html:
                                continue
                            
                            bullet_parts = DocxUtils.clean_html_text(bullet_html)
                            if not any(part[0].strip() for part in bullet_parts):
                                continue

                            bullet_para = right_cell.add_paragraph()
                            DocxUtils.add_word_optimized_spacing(bullet_para, space_after=3)
                            bullet_para.paragraph_format.left_indent = Pt(28)
                            
                            # Add red triangle
                            arrow_run = bullet_para.add_run("▶ ")
                            DocxUtils.add_word_font_optimization(arrow_run, 'Montserrat', 11, True, RGBColor(242, 93, 93))
                            
                            # Add bullet text
                            for text, is_bold in bullet_parts:
                                if text.strip():
                                    bullet_run = bullet_para.add_run(text)
                                    DocxUtils.add_word_font_optimization(bullet_run, 'Montserrat', 10, is_bold, RGBColor(34, 34, 34))
        
        # --- FOOTER (Word-optimized) ---
        footer = doc.sections[0].footer
        footer.is_linked_to_previous = False
        
        # Clear any default footer content
        for para in footer.paragraphs:
            p = para._element
            p.getparent().remove(p)
            
        # Use a table for a full-width colored band
        footer_table = footer.add_table(rows=1, cols=1, width=Inches(8.5))
        footer_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        footer_cell = footer_table.cell(0, 0)
        
        # Set cell background color to orange
        shading_elm = OxmlElement('w:shd')
        shading_elm.set(qn('w:fill'), 'F25D5D')
        footer_cell._tc.get_or_add_tcPr().append(shading_elm)
        
        # Add footer text
        footer_para = footer_cell.paragraphs[0]
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        DocxUtils.add_word_optimized_spacing(footer_para, space_before=4, space_after=4)
        
        footer_run = footer_para.add_run("© www.shorthills.ai")
        DocxUtils.add_word_font_optimization(footer_run, 'Montserrat', 10, False, RGBColor(255, 255, 255))
        
        # Remove any borders from the footer table
        DocxUtils.remove_table_borders(footer_table)
        
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

        # Apply keyword bolding if keywords provided
        data_copy = copy.deepcopy(data) if keywords else data
        
        # Estimate if we need multiple pages
        content_size = estimate_content_size(data_copy)
        needs_multiple_pages = content_size > 65  # Threshold for single page
        
        if not needs_multiple_pages:
            # Use single page layout
            return DocxUtils.generate_docx(data_copy, keywords)
        
        # For multi-page, create a modified single-page layout that flows naturally
        # Word will handle page breaks automatically with proper styling
        doc = docx.Document()
        
        # Set page margins to ZERO for edge-to-edge layout
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0)
            section.bottom_margin = Inches(0)
            section.left_margin = Inches(0)
            section.right_margin = Inches(0)

            # Set header and footer distances
            section.header_distance = Inches(0)
            section.footer_distance = Inches(0)

        # Add page border and background to all pages
        DocxUtils.add_page_border(doc)
        # Skip watermark as requested
        # DocxUtils.add_background_watermark(doc)

        # Header with logos (will appear on all pages)
        header = doc.sections[0].header
        
        # Clear any default header content
        for para in header.paragraphs:
            para.clear()
            
        header_table = header.add_table(rows=1, cols=3, width=docx.shared.Inches(8.0))
        header_table.autofit = False
        header_table.columns[0].width = Inches(2.5)
        header_table.columns[1].width = Inches(3.0)
        header_table.columns[2].width = Inches(2.5)
        
        # Left logo
        left_cell = header_table.cell(0, 0)
        left_para = left_cell.paragraphs[0]
        left_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        left_para.paragraph_format.space_before = Pt(8)
        left_para.paragraph_format.space_after = Pt(4)
        left_para.paragraph_format.left_indent = Pt(8)
        try:
            left_para.add_run().add_picture("templates/left_logo_small.png", height=Inches(0.5))
        except Exception:
            left_run = left_para.add_run("ShorthillsAI")
            left_run.font.name = 'Montserrat'
            left_run.font.size = Pt(12)
            left_run.font.color.rgb = RGBColor(242, 93, 93)
            left_run.bold = True
        
        # Right logo
        right_cell = header_table.cell(0, 2)
        right_para = right_cell.paragraphs[0]
        right_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        right_para.paragraph_format.space_before = Pt(8)
        right_para.paragraph_format.space_after = Pt(4)
        right_para.paragraph_format.right_indent = Pt(8)
        try:
            right_para.add_run().add_picture("templates/right_logo_small.png", height=Inches(0.6))
        except Exception:
            right_run = right_para.add_run("Microsoft Partner")
            right_run.font.name = 'Montserrat'
            right_run.font.size = Pt(10)
            right_run.font.color.rgb = RGBColor(102, 102, 102)

        DocxUtils.remove_table_borders(header_table)

        # For multi-page, use a simplified linear layout instead of complex 2-column
        # This ensures better page breaks and readability
        
        # Add small padding for content
        content_padding = doc.add_paragraph()
        content_padding.paragraph_format.left_indent = Pt(12)
        content_padding.paragraph_format.space_after = Pt(6)

        # Name and title (centered)
        name_para = doc.add_paragraph()
        name_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        name_para.paragraph_format.space_before = Pt(20)
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
            title_parts = DocxUtils.clean_html_text(data_copy['title'])
            for text, is_bold in title_parts:
                if text.strip():
                    title_run = title_para.add_run(text)
                    title_run.font.size = Pt(16)
                    title_run.font.name = 'Montserrat'
                    title_run.font.color.rgb = RGBColor(34, 34, 34)
                    title_run.bold = True

        # Summary
        if data_copy.get('summary'):
            summary_title_para = doc.add_paragraph()
            summary_title_para.paragraph_format.left_indent = Pt(12)
            summary_title_para.paragraph_format.space_before = Pt(0)
            summary_title_para.paragraph_format.space_after = Pt(4)
            summary_title_run = summary_title_para.add_run('SUMMARY')
            summary_title_run.bold = True
            summary_title_run.font.size = Pt(13)
            summary_title_run.font.color.rgb = RGBColor(242, 93, 93)
            summary_title_run.font.name = 'Montserrat'
            
            summary_para = doc.add_paragraph()
            summary_para.paragraph_format.left_indent = Pt(12)
            summary_para.paragraph_format.space_after = Pt(18)
            summary_parts = DocxUtils.clean_html_text(data_copy['summary'])
            DocxUtils.add_formatted_text(summary_para, summary_parts, font_size=12)

        # Projects
        if data_copy.get('projects'):
            projects_title_para = doc.add_paragraph()
            projects_title_para.paragraph_format.left_indent = Pt(12)
            projects_title_para.paragraph_format.space_after = Pt(4)
            projects_title_run = projects_title_para.add_run('KEY RESPONSIBILITIES')
            projects_title_run.bold = True
            projects_title_run.font.size = Pt(13)
            projects_title_run.font.color.rgb = RGBColor(242, 93, 93)
            projects_title_run.font.name = 'Montserrat'
            
            for idx, project in enumerate(data_copy['projects']):
                if project.get('title'):
                    proj_title_para = doc.add_paragraph()
                    proj_title_para.paragraph_format.left_indent = Pt(12)
                    proj_title_para.paragraph_format.space_before = Pt(12)
                    proj_title_para.paragraph_format.space_after = Pt(4)
                    
                    title_run = proj_title_para.add_run(f"Project {idx + 1}: ")
                    title_run.bold = True
                    title_run.font.size = Pt(13)
                    title_run.font.color.rgb = RGBColor(242, 93, 93)
                    title_run.font.name = 'Montserrat'
                    
                    title_parts = DocxUtils.clean_html_text(project['title'])
                    for text, is_bold in title_parts:
                        if text.strip():
                            proj_run = proj_title_para.add_run(text)
                            proj_run.bold = True
                            proj_run.font.size = Pt(13)
                            proj_run.font.color.rgb = RGBColor(242, 93, 93)
                            proj_run.font.name = 'Montserrat'

                if project.get('description'):
                    desc_text = project['description']
                    if isinstance(desc_text, str):
                        # Enhanced bullet splitting to handle HTML lists, newlines, and bullet characters
                        desc_text = desc_text.replace('</li>', '</li>\n').replace('•', '\n')
                        # Split by full stops, newlines, <br>, or <li>.
                        bullets = re.split(r'(?<=[.?!])\s+|\n|<br\s*/?>|<li>', desc_text)
                        
                        for bullet_html in bullets:
                            bullet_html = bullet_html.strip()
                            if not bullet_html:
                                continue
                            
                            bullet_parts = DocxUtils.clean_html_text(bullet_html)

                            if not any(part[0].strip() for part in bullet_parts):
                                continue

                            bullet_para = doc.add_paragraph()
                            bullet_para.paragraph_format.left_indent = Pt(34)  # 12 + 22
                            bullet_para.paragraph_format.space_after = Pt(6)
                            
                            # Add red triangle
                            arrow_run = bullet_para.add_run("▶ ")
                            arrow_run.font.name = 'Montserrat'
                            arrow_run.font.size = Pt(13)
                            arrow_run.font.color.rgb = RGBColor(242, 93, 93)
                            arrow_run.bold = True
                            
                            # Add bullet text
                            DocxUtils.add_formatted_text(bullet_para, bullet_parts, font_size=12)

        # Skills
        if data_copy.get('skills'):
            skills_title_para = doc.add_paragraph()
            skills_title_para.paragraph_format.left_indent = Pt(12)
            skills_title_para.paragraph_format.space_before = Pt(18)
            skills_title_para.paragraph_format.space_after = Pt(4)
            skills_title_run = skills_title_para.add_run('SKILLS')
            skills_title_run.bold = True
            skills_title_run.font.size = Pt(13)
            skills_title_run.font.color.rgb = RGBColor(242, 93, 93)
            skills_title_run.font.name = 'Montserrat'
            
            for skill in data_copy['skills']:
                skill_para = doc.add_paragraph()
                skill_para.paragraph_format.left_indent = Pt(34)  # 12 + 22
                skill_para.paragraph_format.space_after = Pt(6)
                
                # Add red triangle
                arrow_run = skill_para.add_run("▶ ")
                arrow_run.font.name = 'Montserrat'
                arrow_run.font.size = Pt(13)
                arrow_run.font.color.rgb = RGBColor(242, 93, 93)
                arrow_run.bold = True
                
                # Add skill text
                skill_parts = DocxUtils.clean_html_text(skill)
                DocxUtils.add_formatted_text(skill_para, skill_parts, font_size=12)

        # Education
        if data_copy.get('education'):
            edu_title_para = doc.add_paragraph()
            edu_title_para.paragraph_format.left_indent = Pt(12)
            edu_title_para.paragraph_format.space_before = Pt(18)
            edu_title_para.paragraph_format.space_after = Pt(4)
            edu_title_run = edu_title_para.add_run('EDUCATION')
            edu_title_run.bold = True
            edu_title_run.font.size = Pt(13)
            edu_title_run.font.color.rgb = RGBColor(242, 93, 93)
            edu_title_run.font.name = 'Montserrat'
            
            for edu in data_copy['education']:
                if isinstance(edu, dict):
                    para = doc.add_paragraph()
                    para.paragraph_format.left_indent = Pt(34)  # 12 + 22
                    para.paragraph_format.space_after = Pt(6)
                
                    arrow_run = para.add_run("▶ ")
                    arrow_run.font.name = 'Montserrat'
                    arrow_run.font.size = Pt(13)
                    arrow_run.font.color.rgb = RGBColor(242, 93, 93)
                    arrow_run.bold = True
                    
                    if edu.get('degree'):
                        degree_parts = DocxUtils.clean_html_text(edu['degree'])
                        DocxUtils.add_formatted_text(para, degree_parts, font_size=12)
                    
                    if edu.get('institution'):
                        para.add_run('\n')
                        inst_parts = DocxUtils.clean_html_text(edu['institution'])
                        for text, is_bold in inst_parts:
                            if text.strip():
                                inst_run = para.add_run(text)
                                inst_run.font.name = 'Montserrat'
                                inst_run.font.size = Pt(12)
                                inst_run.font.color.rgb = RGBColor(34, 34, 34)
                                inst_run.bold = True
                else:
                    edu_para = doc.add_paragraph()
                    edu_para.paragraph_format.left_indent = Pt(34)
                    edu_para.paragraph_format.space_after = Pt(6)
                    
                    arrow_run = edu_para.add_run("▶ ")
                    arrow_run.font.name = 'Montserrat'
                    arrow_run.font.size = Pt(13)
                    arrow_run.font.color.rgb = RGBColor(242, 93, 93)
                    arrow_run.bold = True
                    
                    edu_parts = DocxUtils.clean_html_text(str(edu))
                    DocxUtils.add_formatted_text(edu_para, edu_parts, font_size=12)

        # Certifications
        if data_copy.get('certifications'):
            cert_title_para = doc.add_paragraph()
            cert_title_para.paragraph_format.left_indent = Pt(12)
            cert_title_para.paragraph_format.space_before = Pt(18)
            cert_title_para.paragraph_format.space_after = Pt(4)
            cert_title_run = cert_title_para.add_run('CERTIFICATIONS')
            cert_title_run.bold = True
            cert_title_run.font.size = Pt(13)
            cert_title_run.font.color.rgb = RGBColor(242, 93, 93)
            cert_title_run.font.name = 'Montserrat'
            
            for cert in data_copy['certifications']:
                para = doc.add_paragraph()
                para.paragraph_format.left_indent = Pt(34)  # 12 + 22
                para.paragraph_format.space_after = Pt(6)
                
                arrow_run = para.add_run("▶ ")
                arrow_run.font.name = 'Montserrat'
                arrow_run.font.size = Pt(13)
                arrow_run.font.color.rgb = RGBColor(242, 93, 93)
                arrow_run.bold = True
                
                if isinstance(cert, dict):
                    if cert.get('title'):
                        title_parts = DocxUtils.clean_html_text(cert['title'])
                        DocxUtils.add_formatted_text(para, title_parts, font_size=12)
                    
                    if cert.get('issuer'):
                        para.add_run('\n')
                        issuer_parts = DocxUtils.clean_html_text(cert['issuer'])
                        for text, is_bold in issuer_parts:
                            if text.strip():
                                issuer_run = para.add_run(text)
                                issuer_run.font.name = 'Montserrat'
                                issuer_run.font.size = Pt(12)
                                issuer_run.font.color.rgb = RGBColor(34, 34, 34)
                                issuer_run.bold = True
                    
                    if cert.get('year'):
                        year_run = para.add_run(f"\n{cert['year']}")
                        year_run.font.name = 'Montserrat'
                        year_run.font.size = Pt(12)
                        year_run.font.color.rgb = RGBColor(34, 34, 34)
                else:
                    cert_parts = DocxUtils.clean_html_text(str(cert))
                    DocxUtils.add_formatted_text(para, cert_parts, font_size=12)

        # Footer that sticks to bottom
        footer = doc.sections[0].footer
        
        # Clear any default footer content
        for para in footer.paragraphs:
            para.clear()
            
        footer_para = footer.add_paragraph()
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        DocxUtils.add_word_optimized_spacing(footer_para, space_before=4, space_after=4)
        
        shading_elm = OxmlElement('w:shd')
        shading_elm.set(qn('w:fill'), 'F25D5D')
        footer_para._p.get_or_add_pPr().append(shading_elm)
        
        footer_run = footer_para.add_run("© www.shorthills.ai")
        DocxUtils.add_word_font_optimization(footer_run, 'Montserrat', 10, False, RGBColor(255, 255, 255))

        # Save document
        docx_file = io.BytesIO()
        doc.save(docx_file)
        docx_file.seek(0)
        return docx_file