### Thomas Middleton

from NewsGroupPreprocessing import GetNewsFileNames, RemoveNewsFormatting
import string
import spacy
import csv


def AddNE(entity, doc_id):
    entity = entity.translate(punc_trans_table)
    entity = entity.lower().strip()
    if len(entity) < 2: return
    if entity not in known_entities:
        known_entities.add(entity)
        entity_doc_dict[entity] = [doc_id]
    else:
        entity_doc_dict[entity].append(doc_id)
    doc_entity_dict[doc_id].append(entity)

def ProcessDocument(model, doc, doc_id):
    proc_doc = model(doc)
    named_entities = [entity.text for entity in proc_doc.ents]
    doc_entity_dict[doc_id] = []
    for NE in named_entities:
        AddNE(NE, doc_id)

def ProcessData(text_data, starting_doc_idx = 0):
    doc_idx = starting_doc_idx
    model = spacy.load("en_core_web_sm")
    for text_n in text_data:
        ProcessDocument(model, text_n, doc_idx)
        doc_idx += 1
    return doc_idx
    
def ExportEntityData(dict_ref, dict_name):
    NG_export_path = "Datasets/Processed/20NG/"
    file_path = NG_export_path + dict_name + ".csv"
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for key, values in dict_ref.items():
            if len(values) > 4:
                row = [key] + values
                writer.writerow(row)

punc_trans_table = str.maketrans("", "", string.punctuation)


db_filenames = GetNewsFileNames()
unformatted_text = RemoveNewsFormatting(db_filenames)

known_entities = set()
entity_doc_dict = {}
doc_entity_dict = {}

num_docs_processed = ProcessData(unformatted_text)
print(num_docs_processed)
ExportEntityData(entity_doc_dict, "EntityDoc")
ExportEntityData(doc_entity_dict, "DocEntity")
