import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

class NERModel:
    def __init__(self, model_name, batch_size=32):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n Using device: {self.device}")
        self.batch_size = batch_size

        # Load the pre-trained model and tokenizer from HuggingFace
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner",
                                     device=self.device,
                                     model=self.model_name,
                                     tokenizer=self.tokenizer,
                                     aggregation_strategy="simple")
        print(f"The {model_name} NER model has been successfully created \n")


    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.ner_pipeline(texts, batch_size=self.batch_size)

    def process_batch(self, batch):
        return self.predict(batch)

    # You can add more methods as needed, for example:
    def preprocess(self, text):
        # Add any preprocessing steps
        return text

    def postprocess(self, entities):
        # Add any postprocessing steps
        return entities
