# core/reporting.py
from pathlib import Path

def ensure_pdf_path(path: str) -> str:
    """Ensures a filename ends with a .pdf extension correctly."""
    p = Path(path)
    # Use with_suffix to correctly replace or add the extension
    return str(p.with_suffix(".pdf"))

def create_pdf_safe(create_pdf_fn, text_content: str, filename: str) -> str:
    """A safe wrapper for the create_pdf_tool."""
    # Ensure content is never empty
    safe_content = (text_content or "").strip() or "No content available for this report."

    # Ensure filename is valid and has the correct extension
    safe_filename = ensure_pdf_path(filename or "report.pdf")

    # Call the original tool function with safe inputs
    return create_pdf_fn(filename=safe_filename, text_content=safe_content)