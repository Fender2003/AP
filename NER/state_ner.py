import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

# 🔹 Step 1: Extended Mapping for Normalization
state_country_map = {
    "new jersey": "nj", "nj": "nj", "newjersey": "nj", "new jersy": "nj", "newjersy": "nj",
    "district of columbia": "dc", "dc": "dc", "d.c.": "dc", "washington dc": "dc", "washington d.c.": "dc",
    "virginia": "va", "va": "va", "virgina": "va", "virginia state": "va",
    "united kingdom": "uk", "uk": "uk", "england": "uk", "u.k.": "uk", "britain": "uk", "great britain": "uk",
    "kansas": "ks", "ks": "ks", "kans": "ks", "kansas state": "ks",
    "texas": "tx", "tx": "tx", "tex": "tx", "tex state": "tx",
    "georgia": "ga", "ga": "ga", "georgya": "ga", "georgie": "ga",
    "illinois": "il", "il": "il", "illinoys": "il", "illinoise": "il",
    "minnesota": "mn", "mn": "mn", "minesotta": "mn", "minnesotta": "mn",
    "new york": "ny", "ny": "ny", "newyork": "ny", "newyork city": "ny", "nyc": "ny",
    "germany": "germany", "deutschland": "germany", "german": "germany", "de": "germany",
    "colorado": "co", "co": "co", "colo": "co", "colorodo": "co",
    "connecticut": "ct", "ct": "ct", "conneticut": "ct", "connecticutt": "ct",
    "ohio": "oh", "oh": "oh", "oheo": "oh", "ohhio": "oh",
    "california": "ca", "ca": "ca", "cali": "ca", "calif": "ca",
    "south kensington": "uk", "s.kensington": "uk", "south ken": "uk",
    "ireland": "ireland", "eiré": "ireland", "eire": "ireland", "irish republic": "ireland",
    "washington": "wa", "wa": "wa", "wash": "wa", "w. state": "wa",
    "hong kong": "hong kong", "hk": "hong kong", "h.k.": "hong kong", "hongkong": "hong kong",
    "peru": "peru", "pru": "peru", "perú": "peru",
    "mexico": "mexico", "mex": "mexico", "méxico": "mexico", "mexican": "mexico",
    "india": "india", "bharat": "india", "ind": "india", "hindustan": "india", "banglore": "india",
    "china": "china", "pud - china": "china", "prc": "china", "chinese": "china",
    "california": "ca", "ca": "ca", "cali": "ca", "calif": "ca",
    "ontario": "ontario", "on": "ontario", "ont": "ontario",
    "kentucky": "ky", "ky": "ky", "kentuckey": "ky", "kty": "ky",
    "russia": "russia", "russian federation": "russia", "russian fed": "russia", "ru": "russia",
    "maryland": "md", "md": "md", "m.d.": "md", "mary land": "md",
    "south africa": "south africa", "sa": "south africa", "s. africa": "south africa",
    "netherlands": "netherlands", "holland": "netherlands", "nl": "netherlands", "nederland": "netherlands",
    "japan": "japan", "jp": "japan", "nippon": "japan",
    "france": "france", "french": "france", "fr": "france",
    "switzerland": "switzerland", "ch": "switzerland", "suisse": "switzerland", "schweiz": "switzerland",
    "italy": "italy", "italia": "italy",
    "south carolina": "sc", "sc": "sc", "s. carolina": "sc",
    "utah": "ut", "ut": "ut", "uta": "ut",
    "canada": "canada", "ca": "canada", "cnd": "canada", "cdn": "canada",
    "pennsylvania": "pa", "pa": "pa", "penn": "pa", "p.a.": "pa",
    "united arab emirates": "uae", "uae": "uae", "u.a.e.": "uae", "emirates": "uae",
    "florida": "fl", "fl": "fl", "fla": "fl",
    "pud - china": "china", "pudong": "china", "pud": "china", "pudong china": "china",
    "brazil": "brazil", "br": "brazil", "brasil": "brazil",
    "sweden": "sweden", "se": "sweden", "sverige": "sweden",
    "australia": "australia", "aus": "australia", "oz": "australia",
    "nicaragua": "nicaragua", "nic": "nicaragua", "nicaragwa": "nicaragua",
    "honduras": "honduras", "hn": "honduras", "honduraz": "honduras",
    "massachusetts": "ma", "ma": "ma", "mass": "ma", "m.a.": "ma",
    "oklahoma": "ok", "ok": "ok", "okla": "ok", "o.k.": "ok",
    "jamaica": "jamaica", "jm": "jamaica",
    "poland": "poland", "pl": "poland", "polska": "poland",
    "czech republic": "czech republic", "cz": "czech republic", "czechia": "czech republic",
    "indonesia": "indonesia", "id": "indonesia",
    "new zealand": "new zealand", "nz": "new zealand", "n. zealand": "new zealand",
    "new york": "ny", "ny": "ny", "n.y.": "ny",
    "michigan": "mi", "mi": "mi", "mich": "mi",
    "tennessee": "tn", "tn": "tn", "tenn": "tn",
    "ontario": "on", "on": "on", "ont": "on",
    "nsw-australia": "nsw", "new south wales": "nsw", "nsw": "nsw",
    "israel": "israel", "il": "israel",
    "cyprus": "cyprus", "cy": "cyprus",
    "north carolina": "nc", "nc": "nc", "n. carolina": "nc",
    "wisconsin": "wi", "wi": "wi", "wisc": "wi",
    "arizona": "az", "az": "az", "ariz": "az",
    "oregon": "or", "ore": "or",
    "romania": "romania", "ro": "romania", "rou": "romania",
    "spain": "spain", "es": "spain", "españa": "spain",
    "colombia": "colombia", "co": "colombia",
    "missouri": "mo", "mo": "mo", "m.o.": "mo", "missoury": "mo",
    "belgium": "belgium", "be": "belgium", "bel": "belgium",
    "taiwan": "taiwan", "tw": "taiwan", "republic of china": "taiwan"
}

# 🔹 Step 2: Create Training Data
def create_training_data(mapping, entity_type="COUNTRY_STATE"):
    return [{"label": entity_type, "pattern": key} for key in mapping.keys()]

# 🔹 Step 3: Generate & Save the NER Model
def generate_country_state_model():
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    patterns = create_training_data(state_country_map, "COUNTRY_STATE")
    ruler.add_patterns(patterns)
    nlp.to_disk("country_state_ner")
    print("✅ Country/State NER model saved.")

# 🔹 Step 4: Normalize Extracted Entities
def normalize_entities(entities):
    return [state_country_map.get(entity.lower(), entity) for entity in entities]

# 🔹 Step 5: Test the Model
def test_model(text):
    nlp = spacy.load("country_state_ner")
    doc = nlp(text)
    extracted = [ent.text for ent in doc.ents]
    normalized = normalize_entities(extracted)

    print("\n🔍 Extracted Entities:", extracted)
    print("✅ Normalized Mapping:", normalized)

# 🔹 Run Everything
if __name__ == "__main__":
    generate_country_state_model()

    # Sample text to test
    with open("state_test.txt", "r") as f:
        text = f.read()
        text = text.lower()

        test_model(text)  # Extract & Normalize
