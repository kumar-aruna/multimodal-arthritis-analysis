"""
Script 10: Generate PDF from Comprehensive Handbook
==================================================

This script converts the markdown handbook to PDF with images included.

Requirements:
- markdown2 (or markdown)
- weasyprint (or pdfkit, or markdown-pdf)
- OR use pandoc (if installed)

Author: Aruna (Bioinformatics Project)
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
HANDBOOK_MD = PROJECT_DIR / "COMPREHENSIVE_PROJECT_HANDBOOK.md"
HANDBOOK_PDF = PROJECT_DIR / "COMPREHENSIVE_PROJECT_HANDBOOK.pdf"

def check_pandoc():
    """Check if pandoc is installed."""
    try:
        result = subprocess.run(['pandoc', '--version'], 
                               capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def generate_pdf_pandoc():
    """Generate PDF using pandoc (recommended method)."""
    print("=" * 60)
    print("Generating PDF using Pandoc")
    print("=" * 60)
    
    # Pandoc command with options
    cmd = [
        'pandoc',
        str(HANDBOOK_MD),
        '-o', str(HANDBOOK_PDF),
        '--pdf-engine=pdflatex',  # or xelatex for better unicode support
        '--toc',  # Table of contents
        '--toc-depth=3',  # Depth of TOC
        '--number-sections',  # Number sections
        '--highlight-style=tango',  # Code highlighting
        '--geometry=margin=1in',  # Page margins
        '--variable=fontsize:11pt',
        '--variable=documentclass:article',
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ PDF generated successfully: {HANDBOOK_PDF}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating PDF: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Pandoc not found. Please install pandoc:")
        print("  macOS: brew install pandoc")
        print("  Linux: sudo apt-get install pandoc")
        print("  Windows: Download from https://pandoc.org/installing.html")
        return False

def generate_pdf_alternative():
    """Alternative method using markdown2 and weasyprint."""
    print("=" * 60)
    print("Generating PDF using Alternative Method")
    print("=" * 60)
    print("Note: This method may not handle images well.")
    print("Recommended: Use pandoc instead.")
    
    try:
        import markdown2
        from weasyprint import HTML
        
        # Read markdown
        with open(HANDBOOK_MD, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables'])
        
        # Add CSS for better formatting
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; }}
                h3 {{ color: #7f8c8d; }}
                code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        HTML(string=html_doc).write_pdf(str(HANDBOOK_PDF))
        print(f"✓ PDF generated: {HANDBOOK_PDF}")
        return True
        
    except ImportError as e:
        print(f"✗ Missing required packages: {e}")
        print("Install with: pip install markdown2 weasyprint")
        return False

def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("PDF HANDBOOK GENERATOR")
    print("=" * 60)
    
    if not HANDBOOK_MD.exists():
        print(f"✗ Handbook not found: {HANDBOOK_MD}")
        return
    
    # Try pandoc first (best method)
    if check_pandoc():
        success = generate_pdf_pandoc()
    else:
        print("\nPandoc not found. Trying alternative method...")
        success = generate_pdf_alternative()
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"PDF saved to: {HANDBOOK_PDF}")
        print(f"File size: {HANDBOOK_PDF.stat().st_size / 1024 / 1024:.2f} MB")
        print("\nYou can now:")
        print("  1. Open the PDF in any PDF viewer")
        print("  2. Share it with others")
        print("  3. Use it for presentations")
    else:
        print("\n" + "=" * 60)
        print("ALTERNATIVE OPTIONS")
        print("=" * 60)
        print("If PDF generation failed, you can:")
        print("  1. Open the markdown file in a markdown viewer")
        print("  2. Use online converters (markdown to PDF)")
        print("  3. Use VS Code with Markdown PDF extension")
        print("  4. Install pandoc: brew install pandoc (macOS)")

if __name__ == "__main__":
    main()

