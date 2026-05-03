"""
VEDA Text Cleaner.

Provides lightweight post-processing for text extracted by OCR engines and
Gemini. Normalizes line breaks, removes Markdown artefacts, fixes hyphenated
words, merges sentence fragments, and strips OCR noise — producing clean,
human-readable output without heavy NLP dependencies.

This module exists because raw OCR and LLM outputs contain formatting
artefacts that degrade downstream text-to-speech quality.

Leverages: Python re (regular expressions).
"""

import re


class TextCleaner:
    """
    Stateless utility for cleaning OCR and Gemini extracted text.

    All methods are static since no instance state is required. The
    cleaning pipeline uses only regex and string operations for speed.

    Leverages: re module.
    """

    @staticmethod
    def clean_extracted_text(text: str) -> str:
        """
        Apply a multi-step cleanup pipeline to raw extracted text.

        Handles line-break normalization, Markdown removal, hyphenation
        repair, sentence-fragment merging, bracket cleanup, stray
        consonant removal, and whitespace collapsing.

        Returns an empty string for non-string or blank input.

        Leverages: re.sub for pattern-based text transformations.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.replace('\\n', '\n').replace('/n', '\n').replace('\r\n', '\n')

        text = text.replace('**', '')
        text = text.replace('__', '')
        text = re.sub(r'(?m)^#+\s*', '', text)

        text = re.sub(r'(?:^|\s)[^\w\s]{1,2}(?:\s|$)', ' ', text)

        text = re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\1\2', text)
        text = re.sub(r'([a-zA-Z]+)-\s+([a-zA-Z]+)', r'\1\2', text)

        text = re.sub(r'(?<![.!?;:])\n(?![A-Z]|\n)', ' ', text)

        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)

        text = re.sub(r'\b[b-hj-zB-HJ-Z]\b', '', text)

        text = re.sub(r'[ \t]+', ' ', text)

        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


clean_extracted_text = TextCleaner.clean_extracted_text
