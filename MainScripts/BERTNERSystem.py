import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

class NERModel:
    def __init__(self, model_name, batch_size=32):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        self.batch_size = batch_size

        # Load the pre-trained model and tokenizer from HuggingFace
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner",
                                     device=self.device,
                                     model=self.model,
                                     tokenizer=self.tokenizer,
                                     aggregation_strategy="simple")
        print(f"The {model_name} NER model has been successfully created \n")

    def align_tokens_to_text(self, tokens, offset_mapping):
        """
        Aligns the tokenized output to the original text, combining subwords into full words.

        Args:
            text (str): The original input text.
            tokens (list of str): The tokenized output from the tokenizer.
            offset_mapping (list of tuples): A list of tuples where each tuple contains the start and end 
                                              positions of the token in the original text.

        Returns:
            list of tuples: Each tuple contains a token and its start and end positions in the original text.
        """
        aligned_tokens = []
        for token, (start, end) in zip(tokens, offset_mapping):
            if token.startswith("##"):
                # Combine subwords into a single token
                aligned_tokens[-1] = (aligned_tokens[-1][0] + token[2:], aligned_tokens[-1][1], end)
            else:
                # Append new token
                aligned_tokens.append((token, start, end))
        return aligned_tokens

    def post_process_ner_results(self, text, results):
        """
        Post-processes the raw NER results to align them with the original text and reconstruct full words.

        Args:
            text (str): The original input text.
            results (list of dict): The raw NER results returned by the pipeline. Each dict contains:
                - 'entity_group': The entity type.
                - 'score': The confidence score.
                - 'word': The tokenized word (may contain subwords).
                - 'start': The start position of the token in the text.
                - 'end': The end position of the token in the text.

        Returns:
            list of dict: The post-processed NER results with reconstructed full words.
        """
        encoding = self.tokenizer(text, return_offsets_mapping=True)
        tokens = encoding.tokens()
        offset_mapping = encoding["offset_mapping"]
        aligned_tokens = self.align_tokens_to_text(tokens, offset_mapping)
        
        for entity in results:
            entity_start = entity['start']
            entity_end = entity['end']
            reconstructed_word = ''
            current_position = entity_start
            
            # Reconstruct the full word from subwords
            for token, token_start, token_end in aligned_tokens:
                if token_start <= entity_start < token_end or token_start < entity_end <= token_end:
                    if token_start > current_position:
                        reconstructed_word += text[current_position:token_start]
                    reconstructed_word += text[token_start:token_end]
                    current_position = token_end
                    if token_end >= entity_end:
                        break
            
            entity['word'] = reconstructed_word
        
        return results

    def process_batch(self, texts):
        """
        Runs NER predictions on the given texts.

        Args:
            texts (str or list of str): The input text(s) to process. If a single string is provided, it will be converted to a list.

        Returns:
            list of list of dict: The NER results for each input text. Each dict contains:
                - 'entity_group': The entity type.
                - 'score': The confidence score.
                - 'word': The tokenized word (with spaces reconstructed).
                - 'start': The start position of the token in the text.
                - 'end': The end position of the token in the text.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        raw_results = self.ner_pipeline(texts, batch_size=self.batch_size)

        processed_results = []
        for text, raw_result in zip(texts, raw_results):
            processed_result = self.post_process_ner_results(text, raw_result)
            processed_results.append(processed_result)
        return processed_results

