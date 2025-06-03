import json
import re
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

# Define CIR types
cir_types = ['Wavelength', 'EVPL', 'EPL', 'DIA', 'Broadband', 'DarkFiber',
       'E_Line', 'MPLS', 'Microwave']

def extract_cir_types(text):
    """Extract CIR types from text using regex."""
    pattern = r'\b(?:' + '|'.join(re.escape(cir) for cir in cir_types) + r')\b'
    return re.findall(pattern, text, re.IGNORECASE)

if __name__ == "__main__":
    with open("cir_type_test.txt", "r", encoding="utf-8") as f:
        test_text = f.read().lower()  # Convert to lowercase for case-insensitive matching

    extracted_cir_types = extract_cir_types(test_text)
    print("Extracted CIR Types:", extracted_cir_types)
