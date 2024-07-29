import torch
import pandas as pd
import spacy
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import load_dataset
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

logger = logging.getLogger(__name__)

class BERTNERModel():
    """
    For a given text, return these attributes:
        Find the person entities in the text
        Find the organisation entities in the text
        Find the location entities in the text
        Find the miscellaneous entities in the text

    Parameters
    ----------
        text (str): the input text
        model (transformers.modeling_auto.AutoModelForTokenClassification):
            the NER model
        tokenizer (transformers.tokenization_auto.AutoTokenizer): the NER
            tokenizer
        device (torch.device): the device to use

    Attributes
    ----------
        ner_results (list): the raw NER results a list of dictionaries
        ner_df (pd.DataFrame): the NER results as a dataframe
        entities (list): the entities types for reference
        unquie_entities (pd.DataFrame): the unique entities found in the text
    """

    def __init__(self, text, model, tokenizer, device):

        self.model = model
        self.tokenizer = tokenizer
        self.text = text
        self.device = device
        self.aggregation_strategy = 'first'
        self.logger = logging.getLogger(__name__)
        self.ner_results = self.get_ner_results()
        # print(self.ner_results)
        self.ner_df = pd.DataFrame(self.ner_results)
        # print(self.ner_df.head())
        self.entities = ['PER', 'ORG', 'LOC', 'MISC']
        self.unique_entities = self.get_unique_entities(self.ner_df)
        # print(self.unique_entities)

    def get_ner_results(self):
        """
        Get the NER results for a given text

        Parameters
        ----------
            text (str): the input texts

        Returns
        -------
            ner_results (list): the NER results a list of dictionaries

        """
        try:
            ner = pipeline(
                'ner',
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy=self.aggregation_strategy)
            self.logger.info(f"Getting NER results for {self.text[:10]}")
            return ner(self.text)
        except Exception as e:
            self.logger.exception(e)
            return []

    def get_entity_df(self, ner_results):
        """
        Get the entity dataframe for the results
        Parameters
        ----------
            ner_results (list): the NER results a list of dictionaries

        Returns
        -------
            entity_df (pandas.DataFrame): the word and entity dataframe
                                         for that entity

        """
        try:
            return pd.DataFrame(ner_results)
        except Exception as e:
            self.logger.exception(e)
            return pd.DataFrame()

    def get_unique_entities(self, df):
        """
        Get the unique entities in the dataframe
        Parameters
        ----------
            df (pandas.DataFrame): the entity dataframe

        Returns
        -------
            unique_entities (pandas.DataFrame): the unique entities in the dataframe

        """
        try:
            return df.groupby(['entity_group', 'word'], as_index=False).agg({
                'score': 'mean',
                'start': 'unique',
                'end': 'unique'
            })
        except Exception as e:
            self.logger.exception("looks like no entities were found", e)
            return pd.DataFrame()




def xxGetSpacyPredictions(nlp, sentences):
    """
    For a list of sentences:
        generate the predictions from the model
        align these predictions with the original words in the sentence.
    ----------
    Parameters:
    nlp : Spacy model
        The loaded Spacy model.
    sentences : List[str]
        List of sentences to process.
    ----------
    Returns:
    all_predictions : List[List[str]]
        Predictions for each word in each sentence, in IOB2 format.
    """
    all_predictions = []
    
    for sentence in sentences:
        doc = nlp(sentence)
        word_predictions = ['O'] * len(doc)
        
        for ent in doc.ents:
            start_token = ent.start
            end_token = ent.end
            
            for i in range(start_token, end_token):
                if i == start_token:
                    word_predictions[i] = f'B-{ent.label_}'
                else:
                    word_predictions[i] = f'I-{ent.label_}'
        
        all_predictions.append(word_predictions)
    
    return all_predictions

def align_tokens(spacy_tokens, original_tokens):
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

def GetSpacyPredictions(nlp, sentences, original_tokens):
    all_predictions = []
    
    # Map Spacy labels to CoNLL-2003 labels
    label_map = {
        'PERSON': 'PER', 'ORG': 'ORG', 'GPE': 'LOC', 'LOC': 'LOC',
        'PRODUCT': 'MISC', 'WORK_OF_ART': 'MISC', 'LAW': 'MISC', 'LANGUAGE': 'MISC',
        'EVENT': 'MISC', 'NORP': 'MISC'
    }
    
    for i, (sentence, orig_tokens) in enumerate(zip(sentences, original_tokens)):
        doc = nlp(sentence)
        spacy_tokens = [token.text for token in doc]
        alignment = align_tokens(spacy_tokens, orig_tokens)
        
        word_predictions = ['O'] * len(orig_tokens)
        
        for ent in doc.ents:
            if ent.label_ in label_map:
                start_token = alignment.get(ent.start, -1)
                end_token = alignment.get(ent.end - 1, -1) + 1
                if start_token != -1 and end_token != -1:
                    for j in range(start_token, end_token):
                        if j < len(word_predictions):
                            if j == start_token:
                                word_predictions[j] = f'B-{label_map[ent.label_]}'
                            else:
                                word_predictions[j] = f'I-{label_map[ent.label_]}'
        
        all_predictions.append(word_predictions)
    return all_predictions

def GetHuggingFacePredictions(ner_pipeline, sentences):
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
        words = sentence.split()
        token_predictions = ner_pipeline(sentence)
        # Initialize all predictions as 'O' (not an NE)
        word_predictions = ['O'] * len(words)

        for pred in token_predictions:
            start_word_index = len(sentence[:pred['start']].split())
            end_word_index = len(sentence[:pred['end']].split())
            
            for i in range(start_word_index, end_word_index):
                if i < len(word_predictions):
                    if i == start_word_index:
                        word_predictions[i] = 'B-' + pred['entity_group']
                    else:
                        word_predictions[i] = 'I-' + pred['entity_group']
        
        all_predictions.append(word_predictions)
    
    return all_predictions



def EvaluateNERModel(model_name, use_debug_prints = False):
    """
    Evaluate a Named Entity Recognition (NER) model on the CoNLL-2003 dataset.
        -loads a pre-trained model
        -processes the CoNLL-2003 test set
        -makes predictions
        -evaluates the model's performance
    ----------
    Parameters:
    model_name : str
        The name or path of the pre-trained model to evaluate.
    use_debug_prints : bool (default = False)
        Print some sample outputs to validate the code works.
    ----------
    Returns:
    evaluation_message : str
        Precision, recall, and F1-score for each entity type.
    """
    # Load the CoNLL-2003 dataset
    dataset = load_dataset("conll2003", trust_remote_code=True)
    test_dataset = dataset["test"]

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

    # Convert numeric labels to string labels
    id2label = {i: label if label != "O" else "O" for i, label in enumerate(dataset["train"].features["ner_tags"].feature.names)}
    true_labels = [[id2label[label] for label in sentence] for sentence in test_dataset["ner_tags"]]

    # Prepare sentences and get predictions
    test_sentences = [" ".join(tokens) for tokens in test_dataset["tokens"]]
    if model_name == 'en_core_web_sm':
        pred_labels = GetSpacyPredictions(nlp, test_sentences, test_dataset["tokens"])
    else:
        pred_labels = GetHuggingFacePredictions(ner_pipeline, test_sentences)


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
    return classification_report(true_labels, pred_labels, mode='strict', scheme=IOB2)

if __name__ == "__main__":
    model_names = ['en_core_web_sm', 'dslim/bert-base-NER']
    for model_name in model_names:
        print(f"Evaluating model: {model_name}")
        evaluation_message = EvaluateNERModel(model_name)
        print(evaluation_message)
        