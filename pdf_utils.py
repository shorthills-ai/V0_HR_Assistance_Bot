import base64
import io
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import fitz  # PyMuPDF
import copy

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
                     right_logo_path="templates/right_logo_small.png"):
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template(template_path)

        bg_image = PDFUtils.get_base64_image(bg_path)
        left_logo_b64 = PDFUtils.get_base64_image(left_logo_path)
        right_logo_b64 = PDFUtils.get_base64_image(right_logo_path)

        # Split projects
        projects = data.get('projects', [])
        first_two = projects[:2]
        rest = projects[2:]

        # Try to fit first two projects on one page by shrinking font size
        font_size = 13
        min_font_size = 9
        pdf_file = None
        html_out = None
        while font_size >= min_font_size:
            data_copy = copy.deepcopy(data)
            data_copy['projects'] = first_two
            html_out = template.render(
                cv=data_copy,
                bg_image=f"data:image/png;base64,{bg_image}",
                left_logo=f"data:image/png;base64,{left_logo_b64}",
                right_logo=f"data:image/png;base64,{right_logo_b64}",
                font_size=font_size
            )
            pdf_file = io.BytesIO()
            HTML(string=html_out).write_pdf(pdf_file)
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            if doc.page_count == 1:
                break
            font_size -= 1
        # Now render the rest of the projects (if any) at normal font size
        if rest:
            # Render the rest as a new page(s)
            data_copy = copy.deepcopy(data)
            data_copy['projects'] = rest
            html_out_rest = template.render(
                cv=data_copy,
                bg_image=f"data:image/png;base64,{bg_image}",
                left_logo=f"data:image/png;base64,{left_logo_b64}",
                right_logo=f"data:image/png;base64,{right_logo_b64}",
                font_size=13
            )
            # Combine the two HTMLs
            full_html = html_out + html_out_rest
            pdf_file = io.BytesIO()
            HTML(string=full_html).write_pdf(pdf_file)
            return pdf_file, full_html
        else:
            return pdf_file, html_out

 