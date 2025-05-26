import base64
import io
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
 
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
        html_out = template.render(
            cv=data,
            bg_image=f"data:image/png;base64,{bg_image}",
            left_logo=f"data:image/png;base64,{left_logo_b64}",
            right_logo=f"data:image/png;base64,{right_logo_b64}",
        )
 
        pdf_file = io.BytesIO()
        HTML(string=html_out).write_pdf(pdf_file)
        return pdf_file, html_out
 
 