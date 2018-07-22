"""
Tutorial based in the following link
https://medium.com/@felixmohr/using-python-and-conditional-random-fields-for-latin-word-segmentation-416ca7a9e513

Another tutorial from github
https://github.com/FelixMohr/NLP-with-Python/blob/master/CRFs-latin-word-segmenation.ipynb
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen, HTTPError
import pycrfsuite


def main():
    print('Initializing main program')
    base_url = "http://www.thelatinlibrary.com/"

    # Take links for the page
    home_content = urlopen(base_url)
    soup = BeautifulSoup(home_content, "lxml")
    author_page_links = soup.find_all("a")
    author_pages = [ap["href"] for i, ap in enumerate(author_page_links) if i < 49]
    ap_content = list()
    texts = list()

    for ap in author_pages:
        ap_content.append(urlopen(base_url + ap))

    # Find all links
    book_links = list()
    for path, content in zip(author_pages, ap_content):
        author_name = path.split('.')[0]
        ap_soup = BeautifulSoup(content, 'lxml')
        book_links += ([link for link in ap_soup.find_all('a', {'href': True}) if author_name in link['href']])

    print(book_links[0])
    print(len(book_links))

    # Get the first 200 characters from the books
    texts = list()
    num_pages = 200

    # Enumerate keeps a counter of iterators
    for i, bl in enumerate(book_links[:num_pages]):
        print('Getting content ' + str(i + 1) + ' of ' + str(num_pages))
        try:
            content = urlopen(base_url + bl['href']).read()
            texts.append(content)
        except HTTPError as err:
            print('Unable to retrieve ' + bl['href'] + '.')
            continue

    # Splits the texts at periods
    sentences = list()

    for i, text in enumerate(texts):
        print('Document ' + str(i + 1) + ' of ' + str(len(texts)))
        textSoup = BeautifulSoup(text, 'lxml')
        paragraphs = textSoup.find_all('p', attrs={'class':None})
        prepared = (''.join([p.text.strip().lower() for p in paragraphs[1:-1]]))
        for t in prepared.split('.'):
            part = ''.join([c for c in t if c.isalpha() or c.isspace()])
            sentences.append(part.strip())

    # Check if it's ok
    print(sentences[200])

    # Get the information if the char makes the beginning of a new word
    prepared_sentences = list()
    for sentence in sentences:
        lengths = [len(w) for w in sentence.split(' ')]
        positions = []

        next_pos = 0
        for length in lengths:
            next_pos = next_pos + length
            positions.append(next_pos)
        concatenated = sentence.replace(' ', '')

        chars = [c for c in concatenated]
        labels = [0 if not i in positions else 1 for i in enumerate(concatenated)]

        prepared_sentences.append(list(zip(chars, labels)))

    print([d for d in prepared_sentences[200]])

    X = [create_sentence_features(ps) for ps in prepared_sentences[:-10000]]
    y = [create_sentence_labels(ps) for ps in prepared_sentences[:-10000]]

    X_test = [create_sentence_features(ps) for ps in prepared_sentences[-10000:]]
    y_test = [create_sentence_labels(ps) for ps in prepared_sentences[-10000:]]

    # Train CRF
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X, y):
        trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1.0,
            'c2': 1e-3,
            'max_iterations': 60,
            'feature.possible_transitions': True
        })

    trainer.train('latin-text-segmentation.crfsuite')

    tagger = pycrfsuite.Tagger()
    tagger.open('latin-text-segmentation.crfsuite')

    print(segment_sentence(tagger, "dominusadtemplumproperat"))
    print(segment_sentence(tagger, "portapatet"))

    print('Done!')


# Function to prepare n-grams
def create_char_features(sentence, i):
    features = [
        'bias',
        'char=' + sentence[i][0]
    ]

    if i >= 1:
        features.extend([
            'char-1=' + sentence[i - 1][0],
            'char-1:0=' + sentence[i - 1][0] + sentence[i][0],
        ])
    else:
        features.append("BOS")

    if i >= 2:
        features.extend([
            'char-2=' + sentence[i - 2][0],
            'char-2:0=' + sentence[i - 2][0] + sentence[i - 1][0] + sentence[i][0],
            'char-2:-1=' + sentence[i - 2][0] + sentence[i - 1][0],
        ])

    if i >= 3:
        features.extend([
            'char-3:0=' + sentence[i - 3][0] + sentence[i - 2][0] + sentence[i - 1][0] + sentence[i][0],
            'char-3:-1=' + sentence[i - 3][0] + sentence[i - 2][0] + sentence[i - 1][0],
        ])

    if i + 1 < len(sentence):
        features.extend([
            'char+1=' + sentence[i + 1][0],
            'char:+1=' + sentence[i][0] + sentence[i + 1][0],
        ])
    else:
        features.append("EOS")

    if i + 2 < len(sentence):
        features.extend([
            'char+2=' + sentence[i + 2][0],
            'char:+2=' + sentence[i][0] + sentence[i + 1][0] + sentence[i + 2][0],
            'char+1:+2=' + sentence[i + 1][0] + sentence[i + 2][0],
        ])

    if i + 3 < len(sentence):
        features.extend([
            'char:+3=' + sentence[i][0] + sentence[i + 1][0] + sentence[i + 2][0] + sentence[i + 3][0],
            'char+1:+3=' + sentence[i + 1][0] + sentence[i + 2][0] + sentence[i + 3][0],
        ])

    return features


def create_sentence_features(prepared_sentence):
    return [create_char_features(prepared_sentence, i) for i in range(len(prepared_sentence))]


def create_sentence_labels(prepared_sentence):
    return [str(part[1]) for part in prepared_sentence]


# Function to segmente sentence
def segment_sentence(tagger, sentence):
    sent = sentence.replace(" ", "")
    prediction = tagger.tag(create_sentence_features(sent))
    complete = ""
    for i, p in enumerate(prediction):
        if p == "1":
            complete += " " + sent[i]
        else:
            complete += sent[i]
    return complete


if __name__ == '__main__':
    main()

