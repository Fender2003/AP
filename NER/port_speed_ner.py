import json
import random
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

mbps_values = [f"{i} mbps" for i in range(1, 1001)] + [f"{i} mb" for i in range(1, 1001)] + [f"{i}mbps" for i in range(1, 1001)] + [f"{i}mb" for i in range(1, 1001)]
gbps_values = [f"{i} gbps" for i in range(1, 101)] + [f"{i} gb" for i in range(1, 101)] + [f"{i}gbps" for i in range(1, 101)] + [f"{i}gb" for i in range(1, 101)]
all_speed_values = mbps_values + gbps_values

def generate_port_speed_dataset():
    data = []
    for value in all_speed_values:
        data.append(value)
    random.shuffle(data)
    return data

def save_dataset(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def create_training_data(port_speeds, entity_type="PORT_SPEED"):
    patterns = []
    for speed in port_speeds:
        pattern = {
            "label": entity_type,
            "pattern": speed
        }
        patterns.append(pattern)
    return patterns

def generate_rules(patterns, model_name="port_speed_ner"):
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    nlp.to_disk(model_name)

def test_model(model, text):
    doc = model(text)
    results = []
    for ent in doc.ents:
        results.append((ent.text, ent.label_))
    return results

if __name__ == "__main__":
    dataset = generate_port_speed_dataset()
    save_dataset(dataset, "port_speed.json")
    print("Custom dataset with port speeds generated and saved.")

    patterns = create_training_data(all_speed_values)

    generate_rules(patterns)
    print("NER rules generated and model saved.")

    nlp = spacy.load("port_speed_ner")

    with open("port_speed_test.txt", "r", encoding="utf-8") as f:
        test_text = f.read()
        test_text = test_text.lower()
    
    entities = test_model(nlp, test_text)
    print("Extracted Entities:")
    for entity_text, entity_label in entities:
        print(f"Entity: {entity_text}, Label: {entity_label}")
