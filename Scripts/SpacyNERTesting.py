### Thomas Middleton
import spacy
import os
import pandas as pd
from datasets import load_dataset
import csv
import string

# Zip together words and respective labels
def process_dataset(dataset_split):
    data = []
    for example in dataset_split:
        words = example["tokens"]
        labels = example["ner_tags"]
        for word, label in zip(words, labels):
            if label == 0:
                data.append({"word": word, "label": 0})
            else:
                data.append({"word": word, "label": 1})
    return data


def SaveLabelledConnl():
    # Load the CoNNL-2003 NER dataset
    dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)

    # Process train, validation, and test sets
    train_data = process_dataset(dataset["train"])
    val_data = process_dataset(dataset["validation"])
    test_data = process_dataset(dataset["test"])

    # Combine all data in a Dataframe
    all_data = train_data + val_data + test_data
    df = pd.DataFrame(all_data)

    # Export
    export_path = os.path.join('Datasets', 'Processed', 'CONLL', 'CONLL_Labels.csv')
    df.to_csv(export_path, index=False)

    print("Data processing and export completed.")

#SaveLabelledConnl()


translation_table = str.maketrans("", "", string.punctuation)

def CheckPunctuation(text):
    text_without_punctuation = text.translate(translation_table)
    return not text_without_punctuation

# Import data as sentences with NE labels
def ImportFormatData(file_path):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        sentence = ""
        labels = []
        for row in csv_reader:
            # End of Sentence
            if row[0] == ".": 
                data.append((sentence, labels))
                sentence = ""
                labels = []
            # Null token
            elif CheckPunctuation(row[0]):
                pass
            # Fix Label
            elif row[0].isdigit():
                sentence = sentence + row[0]
                labels.append(0)
            # Valid Token
            else:
                sentence = sentence + " " + row[0]
                labels.append(int(row[1]))
    return data

def EvaluateModel(model, dataset):
    tp = 0 
    fp = 0 
    fn = 0  
    
    for text, labels in dataset:
        doc = model(text)
        predicted_labels = [1 if ent.ent_type_ else 0 for ent in doc]
        NEs_found = predicted_labels.count(1)
        true_num_NEs = labels.count(1)
        diff = NEs_found - true_num_NEs
        if diff > 0:
            fp += diff
            tp += true_num_NEs
        else:
            fn -= diff
            tp += NEs_found
        if diff > 3:
            #print(text)
            #print(predicted_labels)
            #print(labels)
            pass
    
    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

# Load the SpaCy model
ner_model = spacy.load("en_core_web_sm")
# Import Dataset
import_path = os.path.join('Datasets', 'Processed', 'CONLL', 'CONLL_Labels.csv')
imported_data = ImportFormatData(import_path)

# Test the Model
evaluation_results = EvaluateModel(ner_model, imported_data)

# Print evaluation results
print(f"Precision: {evaluation_results['precision']:.2f}")
print(f"Recall: {evaluation_results['recall']:.2f}")
print(f"F1-score: {evaluation_results['f1']:.2f}")