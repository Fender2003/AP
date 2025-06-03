import pandas
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import json
import random

number_words = {
    "a": 1, "one": 1, "single": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twentyone": 21, "twenty one": 21,
    "twentytwo": 22, "twenty two": 22,
    "twentythree": 23, "twenty three": 23,
    "twentyfour": 24, "twenty four": 24,
    "twentyfive": 25, "twenty five": 25,
    "twentysix": 26, "twenty six": 26,
    "twentyseven": 27, "twenty seven": 27,
    "twentyeight": 28, "twenty eight": 28,
    "twentynine": 29, "twenty nine": 29,
    "thirty": 30, "thirtyone": 31, "thirty one": 31,
    "thirtytwo": 32, "thirty two": 32,
    "thirtythree": 33, "thirty three": 33,
    "thirtyfour": 34, "thirty four": 34,
    "thirtyfive": 35, "thirty five": 35,
    "thirtysix": 36, "thirty six": 36,
    "thirtyseven": 37, "thirty seven": 37,
    "thirtyeight": 38, "thirty eight": 38,
    "thirtynine": 39, "thirty nine": 39,
    "forty": 40, "fortyone": 41, "forty one": 41,
    "fortytwo": 42, "forty two": 42,
    "fortythree": 43, "forty three": 43,
    "fortyfour": 44, "forty four": 44,
    "fortyfive": 45, "forty five": 45,
    "fortysix": 46, "forty six": 46,
    "fortyseven": 47, "forty seven": 47,
    "fortyeight": 48, "forty eight": 48,
    "fortynine": 49, "forty nine": 49,
    "fifty": 50
}



def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def generate_dataset(file):
    data = []

    temp_unit = load_data(file)
    unit = temp_unit['units']
    print("The total units added in the dataset are: ", len(unit))
    print(unit)

    # Digits version
    for un in unit:
        for num in range(1, 101): 
            data.append(f"{num} {un}")
            data.append(f"{num}{un}")

    # Number-words version
    for un in unit:
        for word in number_words.keys():
            data.append(f"{word} {un}")
            data.append(f"{word}{un}")

    random.shuffle(data)
    print("The type of data DS is ", type(data), len(data))
    data = list(set(data)) # remove duplicates
    return data


# for sample in dataset[:50]:
#     print(sample)

def create_training_data(file, type):
    data = generate_dataset(file)
    print(data[:10])#checking data

    patterns = []
    for item in data:
        pattern = {
            "label": type,
            "pattern": item
        }
        patterns.append(pattern)

    return (patterns)

from spacy.lang.en import English
from spacy.pipeline import EntityRuler

def generate_rules(patterns):
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")  
    print("Patterns being added:", patterns) 
    ruler.add_patterns(patterns)
    nlp.to_disk("term_ner")



def test_model(model, text):
    doc = model(text)
    results = []
    for ent in doc.ents: 
        results.append((ent.text, ent.label_)) 
    return results

patterns = create_training_data("term.json", "DATE")
patterns_1 = create_training_data("")
generate_rules(patterns)

nlp = spacy.load("term_ner")
ie_data={}

with open("term_test.txt", "r") as f:
    text = f.read()
    text = text.lower()
    
    # print(text)
    entities = test_model(nlp, text)
    print("Extracted Entities")
    for entity_text, entity_label in entities:
        print(f"Entity: {entity_text}, Label: {entity_label}")

"""
Things we will need in the prompt so that we can find the predicted MRC:
Term
Port_Speed_Mbps
A and Z city or state
    map city and state(all possibilities)
Provider
Cir_Type(add it later)
"""
#deal with CIR type
#test new data
