import re

def clean_for_tts(text: str) -> str:
    """Clean text for optimal TTS pronunciation."""
    # Replace punctuation that causes weird TTS pauses
    text = text.replace("—", ". ").replace("–", ". ").replace(";", ".").replace(":", ".").replace("...", ".")
    text = text.replace("(", ". ").replace(")", ". ")
    text = re.sub(r'\.{2,}', '.', text)
    
    # Expand common abbreviations
    replacements = {
        "AI": "A.I.", "CEO": "C.E.O.", "GDP": "G.D.P.", "ROI": "R.O.I.",
        "U.S.": "the United States", "US": "the United States",
        "UK": "the United Kingdom", "e.g.": "for example",
        "i.e.": "that is", "etc.": "and so on",
        "vs.": "versus", "Dr.": "Doctor", "Mr.": "Mister"
    }
    for abbr, expansion in replacements.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)
    
    text = text.strip()
    if text and text[-1] not in '.!?':
        text += '.'
    return " ".join(text.split())
