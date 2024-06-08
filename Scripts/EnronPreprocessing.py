### Thomas Middleton

# https://www.kaggle.com/code/ankur561999/data-cleaning-enron-email-dataset
# Code Reference

import pandas as pd
import email
import csv
import re


def TestEmailExtraction(email_):
    # email msg string to an email.message.Message obj
    test_email = email.message_from_string(email_['message'])
    # Extract email body
    print(test_email.get_payload())


def PreProcessEnron():
    email_text = []
    for row in email_df.itertuples():
        email_ = email.message_from_string(row.message)
        email_ = email_.get_payload()
        email_ = re.sub(r'[\n\t\r]', ' ', email_)
        email_ = re.sub(r'\s+', ' ', email_)
        email_text.append(email_)

    idx = 1
    with open(enron_export_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for text in email_text:
            entry = [idx, text]
            writer.writerow(entry)
            idx += 1
    print(f" Enron Preprocessed {idx-1} emails.")





enron_raw_path = "Datasets/Raw/EnronEmails/emails.csv"
enron_export_path = "Datasets/Processed/EnronEmails/emails.csv"

email_df = pd.read_csv(enron_raw_path)
#TestEmailExtraction(email_df.loc[1000])
PreProcessEnron()
