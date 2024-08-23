import os, sys

# Set up paths
ZETTINFDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ZETTINFDIR, 'ZETTLIB'))

import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
from pydantic import BaseModel, Field, PrivateAttr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from tqdm import tqdm
import regex as re
import pandas as pd
from NERParser import SQLiteDataset


class SimplifiedExtractor(BaseModel):
    load_dir: str
    device: str = Field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    threshold: float = 0.13
    label_constraint_th: float = 0.95
    relname2template: dict = {}
    relname2desc: dict = {}

    _model: AutoModelForSeq2SeqLM = PrivateAttr()
    _tokenizer: AutoTokenizer = PrivateAttr()
    _rv_model: SentenceTransformer = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.load_dir).to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.load_dir)
        self._rv_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self._rv_model.eval()

    def get_templated_input(self, template: str) -> str:
        x_idx, y_idx = template.find("[X]"), template.find("[Y]")
        if x_idx < y_idx:
            template = template.replace("[X]", "<extra_id_0>").replace(" [Y]", "<extra_id_1>")
        else:
            template = template.replace(" [X]", "<extra_id_1>").replace("[Y]", "<extra_id_0>")
        return template

    def extract_triplets(self, texts: List[str]) -> List[List[Tuple[str, str, str, float]]]:
        all_triplets = []
        for text in texts:
            context_emb = self._rv_model.encode(text)
            rel_scores = {}
            for rel, desc in self.relname2desc.items():
                rel_emb = self._rv_model.encode(f"{rel}. {desc}")
                rel_scores[rel] = distance.cosine(context_emb, rel_emb)
            
            final_target_labels = [rel for rel, score in rel_scores.items() 
                                   if score < self.label_constraint_th]
            
            triplets = []
            for rel in final_target_labels:
                template = self.relname2template[rel]
                input_text = f"{text}</s>{self.get_templated_input(template)}</s>"
                
                inputs = self._tokenizer(input_text, return_tensors="pt").to(self.device)
                outputs = self._model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    num_return_sequences=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                
                for output in outputs:
                    decoded = self._tokenizer.decode(output, skip_special_tokens=False)
                    parts = re.findall(r'<extra_id_\d+>\s*(.*?)\s*(?=<extra_id_|</s>|<pad>)', decoded)
                    if len(parts) >= 2:
                        head, tail = parts[0].strip(), parts[1].strip()
                        if head in text and tail in text and head != tail:
                            # Calculate confidence (you may need to adjust this)
                            confidence = 1.0 - rel_scores[rel]
                            if confidence > self.threshold:
                                match = re.search(r"\[X\](.*?)\[Y\]|\[Y\](.*?)\[X\]", template)
                                template_extract = match.group(1) or match.group(2)
                                triplets.append((head, template_extract, tail, confidence))
            trip_df = pd.DataFrame(triplets, columns=['Head', 'Template', 'Tail', 'Confidence'])
            trip_df = trip_df.sort_values(by='Confidence', ascending=False)
            trip_df.reset_index(drop=True, inplace=True)
            #print(trip_df)
            all_triplets.append(trip_df)
        return all_triplets

    def load_templates(self, template_file: str):
        with open(template_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split("\t")
                if len(tokens) >= 4:
                    rid, rel, templ, desc = tokens[:4]
                    self.relname2desc[rel] = desc
                    self.relname2template[rel] = templ

    def extract(self, texts: List[str]) -> List[List[Tuple[str, str, str, float]]]:
        self._model.eval()
        with torch.no_grad():
            return self.extract_triplets(texts)
        


def process_sqlite_dataset(db_path: str, model_path: str, template_file: str, triplets_export_path: str, batch_size: int = 32, confidence_threshold: float = -0.2, max_triplets_per_relation: int = 1):
    # Initialize dataset and dataloader
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize Inference Model
    extractor = SimplifiedExtractor(load_dir=model_path)
    extractor.load_templates(template_file)

    #all_results = []
    #one_run = False
    #n_run = 0
    # Use tqdm to wrap the dataloader
    with open(triplets_export_path, 'w',  newline='') as triplets_file:
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            #if n_run > 0: break
            doc_ids, sentences = batch
            """
            sentences = [
                "Albert Einstein was born in Ulm, Germany in 1879. He is best known for developing the theory of relativity. Einstein received the Nobel Prize in Physics in 1921 for his services to theoretical physics. He worked at the Swiss Patent Office in Bern early in his career. Later, he became a professor at the University of Berlin. Einstein immigrated to the United States in 1933 due to the rise of Nazi Germany.",
                "The Eiffel Tower, located in Paris, France, was completed in 1889. It was designed by engineer Gustave Eiffel and his team. Originally built as the entrance arch for the 1889 World's Fair, it has become a global cultural icon of France. The tower stands 324 meters tall and weighs 10,100 tons. It remains the tallest structure in Paris and receives millions of visitors each year.",
                "William Shakespeare, born in Stratford-upon-Avon in 1564, is widely regarded as the greatest writer in the English language. He wrote numerous plays, including 'Romeo and Juliet', 'Hamlet', and 'Macbeth'. Shakespeare was also an actor and part-owner of a playing company called the Lord Chamberlain's Men. His works have been translated into many languages and are still performed regularly around the world. Shakespeare died in 1616 at the age of 52.",
                "The Amazon rainforest, primarily located in Brazil, is the world's largest tropical rainforest. It covers approximately 5.5 million square kilometers and is home to an estimated 390 billion individual trees. The Amazon River, which runs through the forest, is the world's largest river by water volume. This ecosystem is crucial for global climate regulation and biodiversity. Unfortunately, the Amazon faces significant threats from deforestation and climate change.",
                "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company's first product was the Apple I personal computer. Apple revolutionized the smartphone industry with the introduction of the iPhone in 2007. Today, Apple is headquartered in Cupertino, California, and is one of the world's most valuable companies. Tim Cook currently serves as the CEO of Apple, having taken over after Jobs' passing in 2011.",
                "The Great Wall of China is a series of fortifications and walls built across the historical northern borders of ancient Chinese states. Construction of the wall began more than 2,300 years ago and continued for centuries. The most well-known sections were built during the Ming Dynasty (1368-1644). The Great Wall stretches over 21,000 kilometers from east to west of China. It was designated as a UNESCO World Heritage site in 1987.",
                "Marie Curie was a Polish-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two scientific fields. Curie discovered the elements polonium and radium with her husband, Pierre Curie. She founded the Curie Institutes in Paris and Warsaw, which remain major centers of medical research today. Marie Curie died in 1934 due to complications from long-term exposure to radiation.",
                "The Beatles were an English rock band formed in Liverpool in 1960. The group consisted of John Lennon, Paul McCartney, George Harrison, and Ringo Starr. They are widely regarded as the most influential band of all time and were integral to the development of 1960s counterculture. The Beatles' best-selling album is \"Sgt. Pepper's Lonely Hearts Club Band,\" released in 1967. The band broke up in 1970, but their music continues to be popular and influential.",
                "Mount Everest, located in the Mahalangur Himal sub-range of the Himalayas, is Earth's highest mountain above sea level. Its peak is 8,848 meters (29,029 ft) above sea level. Everest is situated on the border between Nepal and Tibet. The first confirmed successful ascent was by Edmund Hillary and Tenzing Norgay in 1953. Climbing Everest has become increasingly popular in recent years, leading to concerns about overcrowding and environmental impact.",
                "The International Space Station (ISS) is a modular space station in low Earth orbit. It is a multinational collaborative project involving five space agencies: NASA (United States), Roscosmos (Russia), JAXA (Japan), ESA (Europe), and CSA (Canada). The ISS serves as a microgravity and space environment research laboratory. It orbits Earth at an average altitude of 400 kilometers and travels at 28,000 kilometers per hour. The space station has been continuously occupied since November 2000.",
                "Leonardo da Vinci, born in 1452 in Vinci, Italy, was a polymath of the Renaissance era. He is best known for his paintings, including the Mona Lisa and The Last Supper. Da Vinci was also an engineer, scientist, theorist, sculptor, and architect. His notebooks reveal a wide range of interests, from anatomy to flying machines. Leonardo died in 1519 in Amboise, France, where he spent his final years under the patronage of King Francis I.",
                "The Great Barrier Reef, located off the coast of Queensland in northeastern Australia, is the world's largest coral reef system. It is composed of over 2,900 individual reefs and 900 islands. The reef is home to countless species of colorful fish, mollusks, and starfish, as well as various types of hard and soft corals. It was selected as a World Heritage Site in 1981. Unfortunately, the Great Barrier Reef is under threat from climate change, pollution, and coastal development.",
                "Wolfgang Amadeus Mozart was born in Salzburg, Austria in 1756. He was a prolific and influential composer of the Classical period. Mozart composed over 600 works, including symphonies, concertos, operas, and chamber music. He began composing at the age of five and performed for European royalty. Despite his success, Mozart struggled financially in his later years. He died in Vienna in 1791 at the young age of 35.",
                "The Taj Mahal, located in Agra, India, is an ivory-white marble mausoleum on the right bank of the river Yamuna. It was commissioned in 1632 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal. The Taj Mahal is regarded as one of the finest examples of Mughal architecture, combining Indian, Persian, and Islamic influences. It attracts millions of visitors each year and was designated as a UNESCO World Heritage site in 1983.",
                "SpaceX, founded by Elon Musk in 2002, is a private American aerospace manufacturer and space transportation services company. Its goal is to reduce space transportation costs and enable the colonization of Mars. SpaceX developed the Falcon 1 and Falcon 9 launch vehicles, both of which were designed to be reusable. In 2012, SpaceX's Dragon spacecraft became the first commercial spacecraft to deliver cargo to and from the International Space Station. The company is headquartered in Hawthorne, California.",
                "The pyramids of Giza are ancient monumental structures located on the outskirts of Cairo, Egypt. The Great Pyramid, built for the Pharaoh Khufu, is the oldest and largest of the three pyramids in the Giza complex. It was built around 2560 BCE and stood 146.5 meters tall when completed. The pyramids were built as tombs for the pharaohs and their consorts during the Old and Middle Kingdom periods. They remain one of the Seven Wonders of the Ancient World and attract millions of tourists annually.",
                "Jane Austen, born in 1775 in Hampshire, England, was an English novelist known for her romantic fiction set among the landed gentry. She wrote six major novels, including \"Pride and Prejudice\" and \"Emma.\" Austen's works critique the novels of sensibility of the second half of the 18th century and are part of the transition to 19th-century literary realism. Her use of irony, humor, and social commentary have earned her a place as one of the most widely read writers in English literature. Austen died in 1817 at the age of 41.",
                "The Louvre Museum, located in Paris, France, is the world's largest art museum and a historic monument. It was originally built as a fortress in the 12th century and became a royal residence in the 16th century. The museum officially opened in 1793 with an exhibition of 537 paintings. Today, the Louvre houses over 38,000 objects from prehistory to the 21st century. Its most famous piece is Leonardo da Vinci's Mona Lisa. The museum receives over 8 million visitors annually.",
                "Nelson Mandela was born in 1918 in Mvezo, South Africa. He was a revolutionary and political leader who served as President of South Africa from 1994 to 1999. Mandela was a key figure in the anti-apartheid movement and spent 27 years in prison for his activism. After his release in 1990, he worked to negotiate an end to apartheid. Mandela received the Nobel Peace Prize in 1993 for his efforts to promote reconciliation in South Africa. He died in 2013 at the age of 95.",
                "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait, the one-mile-wide channel between San Francisco Bay and the Pacific Ocean. It was opened in 1937 and was, until 1964, the longest main span of any suspension bridge in the world. The bridge is painted in a distinctive orange color called \"International Orange.\" It is widely considered one of the most photographed bridges in the world and has been declared one of the modern Wonders of the World by the American Society of Civil Engineers."
            ]
            """
            # Process the batch of sentences
            extracted_triplets_list = extractor.extract(sentences)

            # Write unprocessed triplets to file
            doc_ids = doc_ids.tolist()
            for i, df in enumerate(extracted_triplets_list):
                df['DocID'] = doc_ids[i]
                df.to_csv(triplets_file, header=False, index=False)

            """
            # Combine results with document IDs
            for doc_id, sentence, triplets in zip(doc_ids, sentences, extracted_triplets_list):
                all_results.append((doc_id, sentence, triplets))
            
            
            # Test Print
            for chunk_idx in range(len(all_results)):
                d_idx, raw_sentence, triplets_proc = all_results[chunk_idx]
                print("-=-==-=-=-=-=-=-=-=-=-=-==--=-=")
                print("--------")
                print(f"Chunk: {chunk_idx}")
                print(f"DocID: {d_idx}")
                print(f"Raw Sentence: {raw_sentence}")
                print("Triplets:")
                for row in triplets_proc.itertuples(index=False):
                    print(f"[H]: {row.Head} [R]: {row.Template} [T]: {row.Tail} [C]: {row.Confidence}")
            #one_run = True
            """
            #n_run = n_run + 1
    #return all_results
    return True

if __name__ == "__main__":
    db_path = './Datasets/Database/NewsGroupDB3.db'
    model_path = "./ZETTModel/wrapper/fewrel/unseen_5_seed_2/True/extractor/model/"
    template_file = os.path.join(ZETTINFDIR, 'ZETTLIB', 'templates', 'templates.tsv')

    triplets_export_path = './Datasets/Processed/20NG/NGtriplets.csv'

    results = process_sqlite_dataset(db_path, model_path, template_file, triplets_export_path)
    if results:
        print("-=-=-=-=-=-=-=-=-=- \n   CODE HAS FINISHED \n-=-=-=-=-=-=-=-=-=-")
    """
    # Print results
    for doc_id, triplets in results:
        print(f"\nDocument ID: {doc_id}")
        print(f"Extracted triplets:")
        for j, (head, relation, tail, confidence) in enumerate(triplets, 1):
            print(f"{j}. Head: {head}, Relation: {relation}, Tail: {tail}, Confidence: {confidence:.2f}")
        print(f"Total number of extracted triplets: {len(triplets)}")

    print(f"\nTotal number of processed documents: {len(results)}")
    """