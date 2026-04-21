import re

def clean_extracted_text(text: str) -> str:
    """
    Lightweight cleanup for OCR and Gemini extracted text.
    Uses regex and string replacements to ensure clean, human-readable output
    without the overhead of heavy NLP models.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Normalize line breaks
    # Handles incorrect tokens like "/n" or literal "\n" strings from APIs
    text = text.replace('\\n', '\n').replace('/n', '\n').replace('\r\n', '\n')

    # 2. Remove unwanted formatting symbols (Markdown artifacts and stray symbols)
    text = text.replace('**', '')
    text = text.replace('__', '')
    text = re.sub(r'(?m)^#+\s*', '', text)
    
    # Remove isolated stray punctuation or non-alphanumeric noise tokens
    text = re.sub(r'(?:^|\s)[^\w\s]{1,2}(?:\s|$)', ' ', text)

    # 3. Fix hyphenated line breaks and inline hyphenation
    # e.g., "poten-\ntial" or "poten- tial" -> "potential"
    text = re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\1\2', text)
    text = re.sub(r'([a-zA-Z]+)-\s+([a-zA-Z]+)', r'\1\2', text)

    # 4. Merge fragmented lines into coherent sentences
    # If a line breaks without sentence-ending punctuation and the next line starts with lowercase, merge them.
    # Also merge if it just breaks awkwardly (single newline not following punctuation).
    text = re.sub(r'(?<![.!?;:])\n(?![A-Z]|\n)', ' ', text)

    # 5. Fix empty brackets and excessive spacing
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    # Remove space before common ending punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # 6. Remove stray isolated single consonants (likely OCR noise)
    # Preserves valid single letters like 'a', 'A', 'i', 'I' and digits.
    text = re.sub(r'\b[b-hj-zB-HJ-Z]\b', '', text)

    # Collapse multiple spaces and horizontal tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 7. Trim leading/trailing whitespace and limit consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
