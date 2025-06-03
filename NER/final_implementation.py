import json
import re
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

with open("/Users/dhruv/Dhruv/Apcela/NER/state_country_map.json", "r", encoding='utf-8') as f:
    state_country_map = json.load(f)

circuit_types = ['wavelength', 'evpl', 'epl', 'dia', 'broadband', 'darkfibre',
       'e_line', 'mpls', 'microwave']

def normalize_entities(entities):
    return [state_country_map.get(entity.lower(), entity) for entity in entities]

def test_model(text):
    nlp = spacy.load("/Users/dhruv/Dhruv/Apcela/NER/country_state_ner")
    doc = nlp(text)
    extracted = [ent.text for ent in doc.ents]
    normalized = normalize_entities(extracted)
    return normalized

port_speed_nlp = spacy.load("/Users/dhruv/Dhruv/Apcela/NER/port_speed_ner")
state_city_nlp = spacy.load("/Users/dhruv/Dhruv/Apcela/NER/country_state_ner")
term_nlp = spacy.load("/Users/dhruv/Dhruv/Apcela/NER/term_ner")

def extract_entities(text, model):
    doc = model(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_circuit_types(text):
    text = text.lower()
    pattern = r"\b(?:{})\b".format("|".join(circuit_types))
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list(set(matches)) 

if __name__ == "__main__":
    with open("/Users/dhruv/Dhruv/Apcela/NER/common_text.txt", "r") as f:

        text = f.read()
        text = text.lower()

    states_cities = test_model(text)
    port_speeds = extract_entities(text, port_speed_nlp)
    term = extract_entities(text, term_nlp)
    circuit_types_found = extract_circuit_types(text)

    #print(states)
    extracted_data = {
        "PORT_SPEED": port_speeds,
        "STATES_CITIES": states_cities,
        "TERM": term,
        "CIR_TYPES": circuit_types_found
    }

    with open("/Users/dhruv/Dhruv/Apcela/NER/extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)

    print("Extraction completed! Results saved to extracted_data.json.")
