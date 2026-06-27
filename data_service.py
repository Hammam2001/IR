import re
import string
import ir_datasets
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Uncomment if the environment does not already have the data cached.
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    if text is None:
        return ""
    text = str(text)

    text = URL_PATTERN.sub(" ", text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(processed_tokens)


def load_and_preprocess_sample():
    dataset_name = "beir/quora"
    print(f"Loading data from: {dataset_name} ...")
    dataset = ir_datasets.load(dataset_name)

    print("\nConnection established! Here are the first 3 documents before and after processing:")
    print("=" * 60)

    count = 0
    for doc in dataset.docs_iter():
        original_text = doc.text[:200]
        processed_text = preprocess_text(original_text)

        print(f"Document ID: {doc.doc_id}")
        print(f"Original Text:  {original_text}...")
        print(f"Processed Text: {processed_text}...")
        print("-" * 60)

        count += 1
        if count == 3:
            break


if __name__ == "__main__":
    load_and_preprocess_sample()