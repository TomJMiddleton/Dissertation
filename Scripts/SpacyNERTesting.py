### Thomas Middleton
import spacy
import os
import pandas as pd
from datasets import load_dataset

# Load the SpaCy model
#nlp = spacy.load("en_core_web_sm")


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
    
    # Get the label names from the dataset's features
    label_names = dataset["train"].features["ner_tags"].feature.names

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

SaveLabelledConnl()
