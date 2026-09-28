import logging
import os
import shutil
import pypandoc
import tempfile
import re
from typing import Optional

log = logging.getLogger(__name__)


class DocumentGenerator:
    """Handles document generation and conversion operations."""
    
    def __init__(self):
        """Initialize the document generator."""
        self.ensure_pandoc()
    
    def ensure_pandoc(self):
        """
        Ensure Pandoc is available. 
        - Return True if Pandoc is available, else False.
        """

        # Step 1: Check if pandoc binary exists on system PATH
        if shutil.which("pandoc"):
            try:
                version = pypandoc.get_pandoc_version()
                log.info(f"Pandoc found on PATH. Version: {version}")
                return True
            except Exception:
                log.warning("Pandoc appears on PATH but not usable through pypandoc. Attempting repair.")

        # Step 2: Not found → attempt to download via pypandoc
        log.info("Pandoc not found. Attempting to download via pypandoc...")
        try:
            pypandoc.download_pandoc()
            log.info("Pandoc downloaded successfully.")
        except Exception as e:
            log.error(f"Failed to download Pandoc: {e}")
            return False

        # Step 3: Verify installation again
        try:
            version = pypandoc.get_pandoc_version()
            log.info(f"Pandoc is now installed. Version: {version}")
            return True
        except Exception:
            log.error("Pandoc installation verification failed.")
            return False
    
    def convert_to_markdown(self, input_path: str) -> str:
        """
        Convert a document to Markdown format.
        
        Args:
            input_path (str): Path to the input document
            
        Returns:
            str: Markdown content
        """
        try:
            markdown_content = pypandoc.convert_file(
                input_path,
                'markdown',
                format='auto'
            )
            log.info(f"Successfully converted {input_path} to Markdown")
            return markdown_content
        except Exception as e:
            log.error(f"Failed to convert {input_path} to Markdown: {e}")
            raise
    
    def clean_markdown_text(self, markdown_text: str) -> str:
        """
        Clean and prepare markdown text for PDF generation.
        
        Args:
            markdown_text (str): Raw markdown text
            
        Returns:
            str: Cleaned markdown text
        """
        # Remove any problematic characters or patterns
        markdown_text = re.sub(r'\x00', '', markdown_text)  # Remove null bytes
        markdown_text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', markdown_text)  # Remove control chars
        
        # Ensure proper markdown formatting
        markdown_text = markdown_text.strip()
        
        return markdown_text
    
    def safe_pandoc_markdown(self, markdown_text: str) -> str:
        """
        Make markdown text safe for Pandoc processing.
        
        Args:
            markdown_text (str): Raw markdown text
            
        Returns:
            str: Safe markdown text
        """
        # 1. strip BOM if present
        markdown_text = markdown_text.lstrip("\ufeff")

        # 2. remove any accidental control characters
        markdown_text = ''.join(ch for ch in markdown_text if ch.isprintable() or ch in "\n\t\r")

        # 3. ensure the file begins with a valid ASCII hash
        markdown_text = re.sub(r"^[^\#]*#", "#", markdown_text)

        # 4. prevent YAML aliases: *something or &something
        markdown_text = re.sub(r"(?m)^\s*[\*\&][A-Za-z_]\w*", "", markdown_text)

        # 5. fix horizontal rules that are too short
        markdown_text = re.sub(r"^-{1,2}$", "---", markdown_text)

        # 6. ensure blank lines before lists
        markdown_text = re.sub(r"([^\n])\n(\s*[-*+])", r"\1\n\n\2", markdown_text)

        return markdown_text
    
    def generate_from_markdown(self, markdown_text: str, output_path: str, template_path: str = None):
        """
        Generate a DOCX or PDF file from Markdown text.
        
        Args:
            markdown_text (str): Markdown content to convert.
            output_path (str): Output file path ending in .docx or .pdf
            template_path (str, optional): Path to template file. If None, uses default styling.
        
        Returns:
            str: Path to the generated file, or None if generation failed
        """
        log.info(f"Generating file for markdown content: {markdown_text}")
        # Ensure proper blank lines before lists
        markdown_text = markdown_text.replace("\n-", "\n\n-")

        markdown_text = self.safe_pandoc_markdown(markdown_text)
        log.info(f"Safe Pandoc Markdown: {markdown_text}")
        
        ext = os.path.splitext(output_path)[1].lower()
        if ext not in [".docx", ".pdf"]:
            log.error("Output file must end with .docx or .pdf")
            return None

        try:
            # Prepare extra arguments
            extra_args = ["--standalone"]
            if template_path:
                extra_args.extend(["--reference-doc", template_path])
                log.info(f"Using template: {template_path}")
            
            # For PDF, try different PDF engines in order of preference
            if ext == ".pdf":
                # Try weasyprint first (lighter weight)
                try:
                    import weasyprint
                    extra_args.extend(["--pdf-engine=weasyprint"])
                    log.info("Using weasyprint PDF engine")
                except ImportError:
                    # Fallback to pdflatex
                    extra_args.extend(["--pdf-engine=pdflatex"])
                    log.info("Using pdflatex PDF engine")
            else:
                # For DOCX, no special engine needed
                pass
            
            pypandoc.convert_text(
                markdown_text,
                to=ext[1:],  # 'docx' or 'pdf'
                format="markdown",
                outputfile=output_path,
                extra_args=extra_args
            )
            
            log.info(f"File generated successfully: {output_path}")
            return output_path
        except Exception as e:
            log.error(f"Failed to generate {output_path} from Markdown: {e}")
            return None

    def generate_pdf_from_content(self, pdf_content: str, filename: str = "response.pdf") -> str:
        """
        Generate a PDF file from markdown content.
        
        Args:
            pdf_content (str): Markdown content to convert to PDF
            filename (str): Name for the output PDF file
            
        Returns:
            str: Path to the generated PDF file, or None if failed
        """
        if not pdf_content:
            log.error("PDF content is None")
            return None
            
        try:
            # Clean and process markdown content
            cleaned_content = self.clean_markdown_text(pdf_content)
            safe_content = self.safe_pandoc_markdown(cleaned_content)
            
            # Create temporary markdown file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_md:
                temp_md.write(safe_content)
                temp_md.flush()
                
                # Generate PDF using generate_from_markdown method
                output_dir = tempfile.gettempdir()
                output_path = os.path.join(output_dir, filename)
                
                result = self.generate_from_markdown(
                    markdown_text=safe_content,
                    output_path=output_path
                )
                
                log.info(f"PDF generated successfully: {result}")
                return result
                
        except Exception as e:
            log.error(f"Failed to generate PDF from content: {e}")
            return None


# Global instance for easy access
document_generator = DocumentGenerator()
