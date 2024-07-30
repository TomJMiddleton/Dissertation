import torch
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import load_dataset
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2, IOB1
import os
import time
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns

class EvalIO:
    def __init__(self):
        self.results = {}
        self.summary_table = pd.DataFrame()
    
    def __str__(self):
        if len(self.summary_table.index) == 0:
            self.CreateSummaryTable()
        return self.summary_table.to_string()

    def AddEvaluationData(self, model_name, test_set, report, inference_time):
        if model_name not in self.results:
            self.results[model_name] = {}
        
        parsed_report = self.ParseEvaluationReport(report)
        df = pd.DataFrame.from_dict(parsed_report, orient='index')
        
        self.results[model_name][test_set] = {
            'metrics': df,
            'inference_time': inference_time
        }

    def ParseEvaluationReport(self, report):
        lines = report.split('\n')
        data = {}
        
        for line in lines[2:-3]:  # Skip header and footer
            line = line.strip()
            if line:
                matches = re.match(r'(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)', line)
                if matches:
                    label, precision, recall, f1, support = matches.groups()
                    data[label] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1-score': float(f1),
                        'support': int(support)
                    }
        # Parse average scores
        for line in lines[-3:]:
            line = line.strip()
            if line:
                matches = re.match(r'(\w+\s+\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)', line)
                if matches:
                    label, precision, recall, f1, support = matches.groups()
                    data[label] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1-score': float(f1),
                        'support': int(support)
                    }
        
        return data

    def CreateSummaryTable(self):
        assert len(self.results) > 0, "No results stored in EvalIO"
        rows = []
        for model in self.results:
            for dataset in self.results[model]:
                metrics = self.results[model][dataset]['metrics']
                inference_time = self.results[model][dataset]['inference_time']
                
                row = {
                    'Model': model,
                    'Dataset': dataset,
                    'Precision': metrics.loc['weighted avg', 'precision'],
                    'Recall': metrics.loc['weighted avg', 'recall'],
                    'F1-Score': metrics.loc['weighted avg', 'f1-score'],
                    'Inference Time': inference_time
                }
                rows.append(row)
        self.summary_table = pd.DataFrame(rows)

    def NormalisedPerformanceScores(self):
        if len(self.summary_table.index) == 0:
            self.CreateSummaryTable()
        norm_table = self.summary_table

        norm_table['Normalized_F1'] = (norm_table['F1-Score'] - norm_table['F1-Score'].min()) / (norm_table['F1-Score'].max() - norm_table['F1-Score'].min())
        norm_table['Normalized_Time'] = 1 - (norm_table['Inference Time'] - norm_table['Inference Time'].min()) / (norm_table['Inference Time'].max() - norm_table['Inference Time'].min())
        norm_table['Performance_Score'] = (norm_table['Normalized_F1']*3 + norm_table['Normalized_Time']*2) / 5
        
        norm_table = norm_table[['Model', 'Dataset', 'F1-Score', 'Inference Time', 'Performance_Score']]
        return norm_table.sort_values('Performance_Score', ascending=False)

    def PlotEvaluationData(self):
        if len(self.summary_table.index) == 0:
            self.CreateSummaryTable()

        # Bar plot for F1-Scores
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='F1-Score', hue='Dataset', data=self.summary_table)
        plt.title('Model Comparison - F1-Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Scatter plot for F1-Score vs Inference Time
        plt.figure(figsize=(10, 6))
        for dataset in self.summary_table['Dataset'].unique():
            data = self.summary_table[self.summary_table['Dataset'] == dataset]
            plt.scatter(data['Inference Time'], data['F1-Score'], label=dataset)
            for i, model in enumerate(data['Model']):
                plt.annotate(model, (data['Inference Time'].iloc[i], data['F1-Score'].iloc[i]))
        plt.xlabel('Inference Time')
        plt.ylabel('F1-Score')
        plt.title('F1-Score vs Inference Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def ExportResults(self, filename):
        serializable_results = {}
        for model, test_sets in self.results.items():
            serializable_results[model] = {}
            for test_set, data in test_sets.items():
                serializable_results[model][test_set] = {
                    'metrics': data['metrics'].to_dict(orient='index'),
                    'inference_time': data['inference_time']
                }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def ImportResults(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.results = {}  # Clear existing results
        for model, test_sets in data.items():
            for test_set, results in test_sets.items():
                df = pd.DataFrame.from_dict(results['metrics'], orient='index')
                self.results[model] = self.results.get(model, {})
                self.results[model][test_set] = {
                    'metrics': df,
                    'inference_time': results['inference_time']
                }

def ImportWikigoldDataset(file_path):
    """
    Load the Wikigold NER dataset from a text file.
    ----------
    Parameters:
    file_path : str
        Path to the Wikigold dataset text file.
    ----------
    Returns:
    dataset : dict
        A dictionary containing 'tokens' and 'ner_tags' for the dataset.
    """
    tokens = []
    ner_tags = []
    current_sentence = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == '-DOCSTART- O':
                if current_sentence:
                    tokens.append(current_sentence)
                    ner_tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
            elif line:
                token, tag = line.split(' ')
                current_sentence.append(token)
                current_tags.append(tag)
            else:
                if current_sentence:
                    tokens.append(current_sentence)
                    ner_tags.append(current_tags)
                    current_sentence = []
                    current_tags = []

    if current_sentence:
        tokens.append(current_sentence)
        ner_tags.append(current_tags)

    return {'tokens': tokens, 'ner_tags': ner_tags}

def IOB2toIOB1(tags):
    """
    Convert IOB2 tags to IOB1 format.
    """
    iob1_tags = []
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            if i == 0 or tags[i-1] == 'O' or tags[i-1][2:] != tag[2:]:
                iob1_tags.append(tag.replace('B-', 'I-'))
            else:
                iob1_tags.append(tag)
        else:
            iob1_tags.append(tag)
    return iob1_tags

def AlignTokens(spacy_tokens, original_tokens):
    """
    Align tokens from spaCy's tokenization with the dataset's tokenisation structure
    ----------
    Parameters:
    spacy_tokens : List[str]
        Tokens generated by spaCy's tokenizer.
    original_tokens : List[str]
        Original tokens from the dataset.
    ----------
    Returns:
    alignment : Dict[int, int]
        A dictionary mapping spaCy token indices to original token indices.
    """
    alignment = {}
    spacy_idx = 0
    orig_idx = 0
    while spacy_idx < len(spacy_tokens) and orig_idx < len(original_tokens):
        if spacy_tokens[spacy_idx] == original_tokens[orig_idx]:
            alignment[spacy_idx] = orig_idx
            spacy_idx += 1
            orig_idx += 1
        elif spacy_tokens[spacy_idx].startswith(original_tokens[orig_idx]):
            alignment[spacy_idx] = orig_idx
            spacy_idx += 1
        else:
            orig_idx += 1
    return alignment

def GetSpacyPredictions(nlp, sentences, original_tokens, use_iob1 = False):
    """
    For a list of sentences:
        generate the predictions from the model
        align these predictions with the original words in the sentence.
        convert labels to the IOB format
    ----------
    Parameters:
    nlp : spacy.lang
        Loaded spacy model.
    sentences : List[str]
        List of sentences to process.
    original_tokens : List[List[str]]
        Original tokens for each sentence.
    ----------
    Returns:
    all_predictions : List[List[str]]
        Predictions for each word in each sentence, in IOB2 tagging format.
    """
    all_predictions = []
    
    # Map Spacy labels to CoNLL-2003 labels
    label_map = {
        'PERSON': 'PER', 'ORG': 'ORG', 'GPE': 'LOC', 'LOC': 'LOC',
        'PRODUCT': 'MISC', 'WORK_OF_ART': 'MISC', 'LAW': 'MISC', 'LANGUAGE': 'MISC',
        'EVENT': 'MISC', 'NORP': 'MISC'
    }
    
    for i, (sentence, orig_tokens) in enumerate(zip(sentences, original_tokens)):
        # Process the sentence with spacy
        doc = nlp(sentence)
        spacy_tokens = [token.text for token in doc]
        alignment = AlignTokens(spacy_tokens, orig_tokens)
        # Initialize predictions as 'O' (non-entity)
        word_predictions = ['O'] * len(orig_tokens)
        # Iterate through spacy's named entities and map to IOB2 format
        for ent in doc.ents:
            if ent.label_ in label_map:
                start_token = alignment.get(ent.start, -1)
                end_token = alignment.get(ent.end - 1, -1) + 1
                if start_token != -1 and end_token != -1:
                    for j in range(start_token, end_token):
                        if j < len(word_predictions):
                            if j == start_token:
                                if use_iob1:
                                    word_predictions[j] = f'I-{label_map[ent.label_]}'
                                else:
                                    word_predictions[j] = f'B-{label_map[ent.label_]}'
                            else:
                                word_predictions[j] = f'I-{label_map[ent.label_]}'
        
        all_predictions.append(word_predictions)
    return all_predictions

def GetHuggingFacePredictions(ner_pipeline, sentences, use_iob1 = False):
    """
    For a list of sentences:
        generate the predictions from the model
        align these predictions with the original words in the sentence.
    ----------
    Parameters:
    ner_pipeline : transformers.Pipeline
        The NER pipeline to use for predictions.
    sentences : List[str]
        List of sentences to process.
    ----------
    Returns:
    all_predictions : List[List[str]]
        Predictions for each word in each sentence, in IOB2 taging format.
    """
    all_predictions = []
    
    for sentence in sentences:
        # Process the sentence with HF model
        words = sentence.split()
        token_predictions = ner_pipeline(sentence)
        # Initialize all predictions as 'O' (not an NE)
        word_predictions = ['O'] * len(words)
        # Iterate through model's named entities and map to IOB2 format
        for pred in token_predictions:
            start_word_index = len(sentence[:pred['start']].split())
            end_word_index = len(sentence[:pred['end']].split())
            
            for i in range(start_word_index, end_word_index):
                if i < len(word_predictions):
                    if i == start_word_index:
                        if use_iob1:
                            word_predictions[i] = 'I-' + pred['entity_group']
                        else:
                            word_predictions[i] = 'B-' + pred['entity_group']
                    else:
                        word_predictions[i] = 'I-' + pred['entity_group']
        
        all_predictions.append(word_predictions)
    
    return all_predictions



def EvaluateNERModel(
        model_name,
        dataset_name,
        wikigold_file_path='./Datasets/Raw/WikiGold/wikigold.conll.txt',
        use_debug_prints = False):
    """
    Evaluate a Named Entity Recognition (NER) model on the specified dataset.
        -loads a pre-trained model
        -processes the dataset set
        -makes predictions
        -evaluates the model's performance
    ----------
    Parameters:
    model_name : str
        The name or path of the pre-trained model to evaluate.
    dataset_name : str
        The name of the dataset to use ('conll2003' or 'wikigold').
    wikigold_file_path : str, optional
        Path to the Wikigold dataset text file (required if dataset_name is 'wikigold').
    use_debug_prints : bool (default = False)
        Print some sample outputs to validate the code works.
    ----------
    Returns:
    evaluation_message : str
        Precision, recall, and F1-score for each entity type.
    """
    # Load the specified dataset
    if dataset_name == 'conll2003':
        dataset = load_dataset("conll2003", trust_remote_code=True)
        test_dataset = dataset["test"]
        # Convert numeric labels to string labels
        id2label = {i: label if label != "O" else "O" for i, label in enumerate(dataset["train"].features["ner_tags"].feature.names)}
        true_labels = [[id2label[label] for label in sentence] for sentence in test_dataset["ner_tags"]]
    elif dataset_name == 'wikigold':
        #https://github.com/juand-r/entity-recognition-datasets/blob/master/data/wikigold/CONLL-format/docs/entity-list.txt
        assert os.path.exists(wikigold_file_path), f"WikiGold Dataset is missing: {wikigold_file_path}"
        test_dataset = ImportWikigoldDataset(wikigold_file_path)
        true_labels = test_dataset["ner_tags"]
    else:
        raise ValueError("Invalid dataset name.")

    # Set up the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    if model_name == 'en_core_web_sm':
        # Load Spacy model
        nlp = spacy.load(model_name)
    else:
        # Load the pre-trained model and tokenizer from HuggingFace
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ner_pipeline = pipeline("ner", device=device, model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Prepare sentences and get predictions
    use_iob1 = (dataset_name == 'wikigold')
    test_sentences = [" ".join(tokens) for tokens in test_dataset["tokens"]]

    start_time = time.time()
    if model_name == 'en_core_web_sm':
        pred_labels = GetSpacyPredictions(nlp, test_sentences, test_dataset["tokens"], use_iob1 = use_iob1)
    else:
        pred_labels = GetHuggingFacePredictions(ner_pipeline, test_sentences, use_iob1 = use_iob1)
    end_time = time.time()
    execution_time = end_time - start_time

    if use_debug_prints: # Debugging Help
        print(f"Number of true label sequences: {len(true_labels)}")
        print(f"Number of predicted label sequences: {len(pred_labels)}")

        for i in range(5):
            print(f"\nExample {i+1}:")
            print("Sentence:", test_sentences[i])
            print("True labels:", true_labels[i])
            print("Predicted labels:", pred_labels[i])
            print(f"True labels length: {len(true_labels[i])}")
            print(f"Predicted labels length: {len(pred_labels[i])}")

        # Check if lengths match for all samples
        mismatched = [i for i in range(len(true_labels)) if len(true_labels[i]) != len(pred_labels[i])]
        if mismatched:
            print(f"Mismatched lengths found in {len(mismatched)} samples. First 5 mismatched indices: {mismatched[:5]}")

    # Evaluate and return the classification report
    scheme = IOB1 if use_iob1 else IOB2
    return classification_report(true_labels, pred_labels, mode='strict', scheme=scheme), execution_time

if __name__ == "__main__":
    model_names = ['dslim/bert-base-NER', 'dslim/bert-large-NER', 'huggingface-course/bert-finetuned-ner', '51la5/roberta-large-NER', 'Jean-Baptiste/roberta-large-ner-english']#'en_core_web_sm', 
    dataset_names = ['conll2003', 'wikigold']
    evaluator = EvalIO()
    eval_save_path = './Datasets/Processed/evaluation_results.json'

    if True:
        for model_name in model_names:
            for dataset_name in dataset_names:
                print(f" \n Evaluating model: {model_name} on dataset: {dataset_name} \n")
                evaluation_message, execution_time = EvaluateNERModel(model_name, dataset_name)
                evaluator.AddEvaluationData(model_name, dataset_name, evaluation_message, execution_time)
        evaluator.ExportResults(eval_save_path)
    
    evaluator.ImportResults(eval_save_path)
    #print(evaluator)
    #evaluator.PlotEvaluationData()
    print(evaluator.NormalisedPerformanceScores())
