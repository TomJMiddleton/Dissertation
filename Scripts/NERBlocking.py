### Thomas Middleton

from NewsGroupPreprocessing import GetNewsFileNames, RemoveNewsFormatting
import string
import spacy
import csv
import sys


def AddNE(entity, doc_id):
    entity = entity.translate(PUNCTUATION_TRANSLATION_TABLE)
    entity = entity.lower().strip()
    if len(entity) < 2: return
    if entity not in known_entities:
        known_entities.add(entity)
        entity_doc_dict[entity] = [doc_id]
    else:
        if entity not in doc_entities:
            entity_doc_dict[entity].append(doc_id)
    if entity not in doc_entities:
        doc_entity_dict[doc_id].append(entity)
        doc_entities.add(entity)

def ProcessDocument(model, doc, doc_id):
    max_len_doc = 1000000
    split_doc = [doc[i:i + max_len_doc] for i in range(0, len(doc), max_len_doc)]
    named_entities = []
    for split_ in split_doc:
        proc_doc = model(split_)
        named_entities += [entity.text for entity in proc_doc.ents]
    
    doc_entity_dict[doc_id] = []
    for NE in named_entities:
        AddNE(NE, doc_id)

def ProcessData(text_data, starting_doc_idx = 0):
    doc_idx = starting_doc_idx
    model = spacy.load("en_core_web_sm")
    for text_n in text_data:
        doc_entities.clear()
        ProcessDocument(model, text_n, doc_idx)
        doc_idx += 1
    return doc_idx
    
def ExportEntityData(dict_ref, dict_name, dataset_name):
    NG_export_path = "Datasets/Processed/"
    file_path = NG_export_path + dataset_name + "/" + dict_name + ".csv"
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for key, values in dict_ref.items():
            if len(values) > 4:
                row = [key] + values
                writer.writerow(row)

def NERForNewsGroup():
    known_entities.clear()
    doc_entities.clear()
    entity_doc_dict.clear()
    doc_entity_dict.clear()

    db_filenames = GetNewsFileNames()
    unformatted_text = RemoveNewsFormatting(db_filenames)
    num_docs_processed = ProcessData(unformatted_text)
    print(f"\n Number of documents Name Entity Processed: {num_docs_processed} \n")
    ExportEntityData(entity_doc_dict, "EntityDoc", "20NG")
    ExportEntityData(doc_entity_dict, "DocEntity", "20NG")
    print("\n Data Exported \n")

def RetrieveEnron():
    enron_path = "Datasets/Processed/EnronEmails/emails.csv"
    enron_texts = []

    max_int = sys.maxsize
    while True:
        # Decrease max_int until it works with field_size_limit
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
    print(f"field size limit found: {max_int}")
    with open(enron_path, mode='r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            enron_texts.append(row[1])
    return enron_texts

def NERForEnron():
    known_entities.clear()
    doc_entities.clear()
    entity_doc_dict.clear()
    doc_entity_dict.clear()

    unformatted_text = RetrieveEnron()
    print("All Enron Read in")
    num_docs_processed = ProcessData(unformatted_text)
    print(f"\n Number of documents Name Entity Processed: {num_docs_processed} \n")
    ExportEntityData(entity_doc_dict, "EntityDoc", "EnronEmails")
    ExportEntityData(doc_entity_dict, "DocEntity", "EnronEmails")
    print("\n Data Exported \n")


PUNCTUATION_TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)
known_entities = set()
doc_entities = set()
entity_doc_dict = {}
doc_entity_dict = {}

#NERForNewsGroup()
NERForEnron()