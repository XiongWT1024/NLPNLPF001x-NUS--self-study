import string
import random
from collections import Counter

from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from datasets import load_dataset

nltk_stopwords = list(stopwords.words('english'))
punctuation_translator = str.maketrans('', '', string.punctuation)

# The overall label set for the original go_emotions dataset.
go_emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]

# In this assignment, we only consider the following five labels.
selected_labels = ["approval", "joy", "anger", "sadness", "confusion"]


def prepare_corpus():
    """
    Get go_emotions dataset from Huggingface Datasets and simplify it.
    """

    # get go_emotion dataset
    go_emotions_dataset = load_dataset("go_emotions", cache_dir="./data/")

    def filter_dataset(split):
        """
        # Preprocess
        # (1) We only consider instances with one emotion label.
        # (2) We select instances labeled with "approval", "joy", "anger", "sadness" and "confusion" as our assignment dataset.
        """
        dataset = {}

        # pre-process
        for data in go_emotions_dataset[split]:
            if len(data["labels"]) == 1 and go_emotion_labels[data["labels"][0]] in selected_labels:
                label = go_emotion_labels[data["labels"][0]]
                if label in dataset:
                    dataset[label].append((data["text"], label))
                else:
                    dataset[label] = [(data["text"], label)]

        # shuffle dataset
        instances = []
        for label in selected_labels:
            for data in dataset[label]:
                instances.append((data[0].strip(), data[1].strip()))
        random.shuffle(instances)

        # post-process
        texts = []
        labels = []
        for instance in instances:
            texts.append(instance[0])
            labels.append(instance[1])

        # show information
        print(f"go_emotions dataset [{split}] part statistics: ", dict(Counter(labels)))

        return texts, labels

    train_texts, train_labels = filter_dataset("train")
    print("Train dataset size: ", len(train_texts))

    test_texts, test_labels = filter_dataset("test")
    print("Text dataset size: ", len(test_texts))
    return train_texts, train_labels, test_texts, test_labels


def lemmatize_token_list(lemmatizer, token_list):
    pos_tag_list = pos_tag(token_list)
    for idx, (token, tag) in enumerate(pos_tag_list):
        tag_simple = tag[0].lower()  # Converts, e.g., "VBD" to "c"
        if tag_simple in ['n', 'v', 'j']:
            word_type = tag_simple.replace('j', 'a')
        else:
            word_type = 'n'
        lemmatized_token = lemmatizer.lemmatize(token, pos=word_type)
        token_list[idx] = lemmatized_token
    return token_list


def preprocess_text(s, tokenizer=None, remove_stopwords=True, remove_punctuation=True,
                    lemmatizer=None, lowercase=True, return_type='str'):
    """
    Preprocess one text
    """
    # Tokenization either with default tokenizer or user-specified tokenizer
    if tokenizer is None:
        token_list = word_tokenize(s)
    else:
        token_list = tokenizer.tokenize(s)

    # Stem or lemmatize if needed
    if lemmatizer is not None:
        token_list = lemmatize_token_list(lemmatizer, token_list)

    # Convert all tokens to lowercase if need
    if lowercase:
        token_list = [token.lower() for token in token_list]

    # Remove all stopwords if needed
    if remove_stopwords:
        token_list = [token for token in token_list if not token in nltk_stopwords]

    # Remove all punctuation marks if needed (note: also converts, e.g, "Mr." to "Mr")
    if remove_punctuation:
        token_list = [''.join(c for c in s if (c not in string.punctuation and c != "’")) for s in token_list]
        token_list = [token for token in token_list if len(token) > 0]  # Remove "empty" tokens

    return token_list


def preprocess_texts(texts):
    """
    Preprocess texts, including `Lemmatization`, `Stop words Removal`, `Punctuation Removal` and `Lowercase`.
    [Input] texts (list of string)
    """
    processed_texts = []

    wordnet_lemmatizer = WordNetLemmatizer()

    for text in texts:
        processed_text = preprocess_text(text, lemmatizer=wordnet_lemmatizer)
        processed_texts.append(processed_text)

    assert len(processed_texts) == len(texts)
    return processed_texts