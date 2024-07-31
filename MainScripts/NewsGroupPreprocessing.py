### Thomas Middleton
import os
import re
import pandas as pd

# Regular expressions to match the required information
newsgroup_re = re.compile(r'^Newsgroup:\s*(.*)', re.MULTILINE)
document_id_re = re.compile(r'^document_id:\s*(\d+)', re.MULTILINE)
document_re = re.compile(r'^Newsgroup:.*?(?=^Newsgroup:|\Z)', re.DOTALL | re.MULTILINE)
subject_re = re.compile(r'^Subject:.*', re.MULTILINE)

# Dataset filepath
data_path = "./Datasets/Raw/20NG/"

# Get the filenames of the dataset
def GetNewsFileNames():
    dir_names = os.listdir(data_path)
    return [file for file in dir_names if file.endswith('.txt')]

def ExtractDocumentIndex(document):
    newsgroup = newsgroup_re.search(document).group(1) if newsgroup_re.search(document) else None
    document_id = document_id_re.search(document).group(1) if document_id_re.search(document) else None
    return newsgroup, document_id

def ExtractDocumentText(document):
    lines = document.split('\n')
    main_text_lines = lines[4:]
    text = '\n'.join(main_text_lines).strip()
    #text = re.sub(r'\s+', ' ', text).strip()
    return text

def GenerateNewsIndexes(news_filenames):
    newsgroup_idx = []
    doc_idx = []
    for filename in news_filenames:
        with open(os.path.join(data_path, filename), 'r') as file:
            content = file.read()
            documents = document_re.findall(content)
            for doc in documents:
                newsgroup, document_id = ExtractDocumentIndex(doc)
                if newsgroup and document_id:
                    newsgroup_idx.append(newsgroup)
                    doc_idx.append(document_id)
    return {'newsgroup':newsgroup_idx,'document_id':doc_idx}

def GenerateNewsIndexFile(output_path):
    db_filenames = GetNewsFileNames()
    db_indexes = GenerateNewsIndexes(db_filenames)
    news_index_df = pd.DataFrame(db_indexes)
    output_filename = "NewsGroupIndex.csv"
    news_index_df.to_csv(os.path.join(output_path, output_filename), index=False)

# GenerateNewsIndexFile(data_path)

def RemoveNewsFormatting(news_filenames):
    newsgroup_text = []
    for filename in news_filenames:
        with open(os.path.join(data_path, filename), 'r') as file:
            content = file.read()
            documents = document_re.findall(content)
            for doc in documents:
                newsgroup, document_id = ExtractDocumentIndex(doc)
                if newsgroup and document_id:
                    newsgroup_text.append(ExtractDocumentText(doc))
    return newsgroup_text

def CleanNewsGroupBasics(text):
    # Remove XML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Split into lines
    lines = text.split('\n')
    cleaned_lines = []
    current_paragraph = []

    # Metadata patterns to remove
    metadata_patterns = [
        r'^Archive-name:',
        r'^Alt-atheism-archive-name:',
        r'^Last-modified:',
        r'^Version:'
    ]

    for line in lines:
        line = line.strip()
        
        # Skip metadata lines
        if any(re.match(pattern, line) for pattern in metadata_patterns):
            continue

        if line:
            if line.isupper() and len(line) > 3:  # Likely a header
                if current_paragraph:
                    cleaned_lines.append(' '.join(current_paragraph))
                    cleaned_lines.append('')  # Add a blank line after paragraph
                    current_paragraph = []
                cleaned_lines.append(line)  # Keep header as is
            else:
                current_paragraph.append(line)
        elif current_paragraph:
            cleaned_lines.append(' '.join(current_paragraph))
            cleaned_lines.append('')  # Add a blank line after paragraph
            current_paragraph = []
        else:
            # If we encounter an empty line and we're not in a paragraph, just add it
            cleaned_lines.append('')
    
    if current_paragraph:
        cleaned_lines.append(' '.join(current_paragraph))

    # Join lines
    cleaned_text = '\n'.join(cleaned_lines)

    # Remove multiple consecutive blank lines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    # Remove special characters (non-ASCII)
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)

    # Remove some irregular <, >, | characters
    cleaned_text = re.sub(r'[|<>]', '', cleaned_text)

    return cleaned_text.strip()

def TestNewsCleaning():
    db_filenames = GetNewsFileNames()
    unformatted_text = RemoveNewsFormatting(db_filenames)
    #print(len(unformatted_text))
    #print(unformatted_text[7])

TestNewsCleaning()