import io
import logging
from typing import List

class PDFParser:
    """
    A parser to extract text from PDF content.
    """
    def __init__(self):
        pass

    def parse(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes.
        Tries pypdf first (modern), then falls back to pdfminer.six.
        """
        if not pdf_bytes:
            return ""

        # Try pypdf (modern)
        try:
            import pypdf
            with io.BytesIO(pdf_bytes) as bio:
                reader = pypdf.PdfReader(bio)
                text_parts = []
                for page in reader.pages:
                    part = page.extract_text()
                    if part:
                        text_parts.append(part)
                text = "\n".join(text_parts)
                if text.strip():
                    return text.strip()
        except ImportError:
            pass
        except Exception as e:
            logging.error(f"pypdf extraction failed: {e}")

        # Try pdfminer.six (fallback)
        try:
            from pdfminer.high_level import extract_text
            with io.BytesIO(pdf_bytes) as bio:
                text = extract_text(bio) or ""
            if text.strip():
                return text.strip()
        except ImportError:
            pass
        except Exception as e:
            logging.error(f"pdfminer extraction failed: {e}")

        # Try PyPDF2 (old fallback)
        try:
            import PyPDF2
            with io.BytesIO(pdf_bytes) as bio:
                reader = PyPDF2.PdfReader(bio)
                text_parts = []
                # In PyPDF2, it might be reader.pages or reader.getNumPages()
                pages = getattr(reader, "pages", [])
                for page in pages:
                    part = page.extract_text()
                    if part:
                        text_parts.append(part)
                text = "\n".join(text_parts)
                if text.strip():
                    return text.strip()
        except ImportError:
            pass
        except Exception as e:
            logging.error(f"PyPDF2 extraction failed: {e}")

        return ""

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Convenience function for backward compatibility.
    """
    parser = PDFParser()
    return parser.parse(pdf_bytes)
