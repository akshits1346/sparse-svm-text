from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    dataset = fetch_20newsgroups(
        subset='train',
        categories=['rec.sport.hockey', 'sci.space'],
        remove=('headers', 'footers', 'quotes')
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )

    X = vectorizer.fit_transform(dataset.data)
    y = dataset.target

    return X, y


