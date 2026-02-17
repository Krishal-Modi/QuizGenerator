"""
Document Service - Extract text from PDF, TXT, DOCX files
"""
import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DocumentSection:
    """Represents a section of the document"""
    title: str
    content: str
    level: int  # Heading level (1, 2, 3, etc.)
    start_idx: int
    end_idx: int


class DocumentService:
    """
    Service for extracting and processing text from lecture notes.
    Supports PDF, TXT, and DOCX formats.
    """
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.txt', '.docx'}
    
    def extract_text(self, filepath: str) -> str:
        """
        Extract text from document based on file type.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Extracted text content
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.pdf':
            return self._extract_from_pdf(filepath)
        elif ext == '.txt':
            return self._extract_from_txt(filepath)
        elif ext == '.docx':
            return self._extract_from_docx(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _extract_from_pdf(self, filepath: str) -> str:
        """Extract text from PDF file using pdfplumber - reads ALL pages thoroughly"""
        try:
            import pdfplumber
            
            text_parts = []
            total_pages = 0
            extracted_pages = 0
            
            with pdfplumber.open(filepath) as pdf:
                total_pages = len(pdf.pages)
                print(f"[PDF] Opening PDF with {total_pages} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = ''
                    
                    # Method 1: Standard text extraction
                    try:
                        extracted = page.extract_text(
                            x_tolerance=3,
                            y_tolerance=3,
                            layout=False
                        )
                        if extracted:
                            page_text = extracted
                    except Exception as e:
                        print(f"[PDF] Standard extraction failed for page {page_num}: {e}")
                    
                    # Method 2: Try layout-based extraction if standard got little text
                    if len(page_text.strip()) < 50:
                        try:
                            extracted = page.extract_text(layout=True)
                            if extracted and len(extracted.strip()) > len(page_text.strip()):
                                page_text = extracted
                        except Exception:
                            pass
                    
                    # Method 3: Extract from tables if present
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                for row in table:
                                    if row:
                                        row_text = ' | '.join([str(cell) if cell else '' for cell in row])
                                        if row_text.strip() and row_text.strip() not in page_text:
                                            page_text += '\n' + row_text
                    except Exception:
                        pass
                    
                    if page_text and page_text.strip():
                        page_text = self._clean_pdf_text(page_text)
                        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                        extracted_pages += 1
            
            full_text = '\n\n'.join(text_parts)
            print(f"[PDF] Extracted text from {extracted_pages}/{total_pages} pages ({len(full_text)} chars)")
            
            if not full_text.strip():
                print("[PDF] pdfplumber got no text, trying PyMuPDF fallback")
                return self._extract_from_pdf_pymupdf(filepath)
            
            return full_text
            
        except ImportError:
            return self._extract_from_pdf_pymupdf(filepath)
        except Exception as e:
            print(f"[PDF] pdfplumber failed: {e}, trying PyMuPDF")
            return self._extract_from_pdf_pymupdf(filepath)
    
    def _extract_from_pdf_pymupdf(self, filepath: str) -> str:
        """Fallback PDF extraction using PyMuPDF - reads ALL pages"""
        try:
            import fitz  # PyMuPDF
            
            text_parts = []
            
            with fitz.open(filepath) as doc:
                total_pages = len(doc)
                print(f"[PDF-PyMuPDF] Opening PDF with {total_pages} pages")
                
                for page_num, page in enumerate(doc, 1):
                    # Try multiple extraction methods
                    text = page.get_text("text")
                    
                    # If standard text extraction gets little, try blocks
                    if len(text.strip()) < 50:
                        blocks = page.get_text("blocks")
                        if blocks:
                            block_texts = [b[4] for b in blocks if b[6] == 0]  # text blocks only
                            alt_text = '\n'.join(block_texts)
                            if len(alt_text.strip()) > len(text.strip()):
                                text = alt_text
                    
                    if text and text.strip():
                        text = self._clean_pdf_text(text)
                        text_parts.append(f"--- Page {page_num} ---\n{text}")
            
            full_text = '\n\n'.join(text_parts)
            print(f"[PDF-PyMuPDF] Extracted {len(full_text)} chars from {total_pages} pages")
            return full_text
            
        except ImportError:
            raise ImportError("Neither pdfplumber nor PyMuPDF installed. "
                            "Install with: pip install pdfplumber or pip install PyMuPDF")
    
    def _extract_from_txt(self, filepath: str) -> str:
        """Extract text from plain text file"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Unable to decode text file with common encodings")
    
    def _extract_from_docx(self, filepath: str) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            
            doc = Document(filepath)
            
            text_parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            return '\n\n'.join(text_parts)
            
        except ImportError:
            raise ImportError("python-docx not installed. "
                            "Install with: pip install python-docx")
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean common PDF extraction artifacts"""
        # Remove page numbers (standalone numbers)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove header/footer patterns (customize based on your documents)
        text = re.sub(r'^Page \d+ of \d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix hyphenated words split across lines
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        return text.strip()
    
    def segment_text(self, text: str) -> List[DocumentSection]:
        """
        Segment text into sections based on headings.
        
        Args:
            text: Full document text
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        
        # Common heading patterns
        heading_patterns = [
            (r'^#{1,6}\s+(.+)$', 'markdown'),  # Markdown headings
            (r'^(\d+\.)+\s+(.+)$', 'numbered'),  # Numbered headings (1.1, 1.2.3)
            (r'^[A-Z][A-Z\s]+$', 'caps'),  # ALL CAPS headings
            (r'^[IVX]+\.\s+(.+)$', 'roman'),  # Roman numeral headings
        ]
        
        lines = text.split('\n')
        current_section = {'title': 'Introduction', 'content': [], 'level': 1, 'start': 0}
        
        for i, line in enumerate(lines):
            is_heading = False
            heading_level = 1
            heading_text = line.strip()
            
            # Check for heading patterns
            for pattern, pattern_type in heading_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    is_heading = True
                    if pattern_type == 'markdown':
                        heading_level = line.count('#')
                        heading_text = match.group(1)
                    elif pattern_type == 'numbered':
                        heading_level = line.count('.')
                        heading_text = match.group(0)
                    break
            
            # Also check for short lines followed by content (potential headings)
            if not is_heading and len(line.strip()) < 100 and line.strip():
                # Check if next line is blank or starts new paragraph
                if i + 1 < len(lines) and (not lines[i + 1].strip() or 
                    lines[i + 1].strip() and not lines[i + 1].startswith(' ')):
                    # Could be a heading - use heuristics
                    if re.match(r'^[A-Z][\w\s]+$', line.strip()):
                        is_heading = True
            
            if is_heading and current_section['content']:
                # Save current section
                sections.append(DocumentSection(
                    title=current_section['title'],
                    content='\n'.join(current_section['content']),
                    level=current_section['level'],
                    start_idx=current_section['start'],
                    end_idx=i - 1
                ))
                
                # Start new section
                current_section = {
                    'title': heading_text,
                    'content': [],
                    'level': heading_level,
                    'start': i
                }
            else:
                current_section['content'].append(line)
        
        # Add last section
        if current_section['content']:
            sections.append(DocumentSection(
                title=current_section['title'],
                content='\n'.join(current_section['content']),
                level=current_section['level'],
                start_idx=current_section['start'],
                end_idx=len(lines) - 1
            ))
        
        return sections
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text for question generation.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            
            sentences = sent_tokenize(text)
            
        except ImportError:
            # Fallback to simple regex-based sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short or very long sentences
        sentences = [s.strip() for s in sentences 
                    if 20 < len(s.strip()) < 500]
        
        return sentences
    
    def get_context_for_concept(self, text: str, concept: str, 
                                 context_size: int = 500) -> str:
        """
        Get surrounding context for a concept in the text.
        
        Args:
            text: Full document text
            concept: Concept to find context for
            context_size: Number of characters to include around concept
            
        Returns:
            Context string containing the concept
        """
        # Find concept in text (case-insensitive)
        pattern = re.compile(re.escape(concept), re.IGNORECASE)
        match = pattern.search(text)
        
        if not match:
            return ""
        
        start = max(0, match.start() - context_size // 2)
        end = min(len(text), match.end() + context_size // 2)
        
        context = text[start:end]
        
        # Clean up to complete sentences
        first_period = context.find('. ')
        if first_period > 0 and first_period < 50:
            context = context[first_period + 2:]
        
        last_period = context.rfind('. ')
        if last_period > len(context) - 50:
            context = context[:last_period + 1]
        
        return context.strip()
    
    def get_document_stats(self, text: str) -> Dict:
        """
        Get statistics about the document.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with document statistics
        """
        words = text.split()
        sentences = self.extract_sentences(text)
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'character_count': len(text)
        }
