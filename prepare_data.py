"""Download the Cornell Movie Dialogs Corpus and turn it into chat.txt.

The corpus: 220,579 conversational exchanges between 10,292 movie characters
across 617 movies. Official source and citation:
  https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
  Cristian Danescu-Niculescu-Mizil & Lillian Lee,
  "Chameleons in imagined conversations", ACL 2011.

Every consecutive pair of lines becomes one tiny conversation:

    User: <first line>
    Bot: <second line>

Exchanges are separated by a blank line. The GPT learns the pattern
"after `User: ...` and `Bot:` comes a reply" - which is all chat.py needs.

  python prepare_data.py          # -> chat.txt  (~9 MB)
"""
import os, re, zipfile, random, urllib.request

URL = 'https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
ZIP = 'cornell_movie_dialogs_corpus.zip'
SEP = ' +++$+++ '
MAX_PAIRS = 120_000      # ~9 MB of text: a couple of epochs in an evening of training
MAX_LINE = 200           # longer lines waste the context window


def main():
    if not os.path.exists(ZIP):
        print('downloading corpus (~10 MB) ...')
        urllib.request.urlretrieve(URL, ZIP)
    z = zipfile.ZipFile(ZIP)

    lines = {}
    with z.open('cornell movie-dialogs corpus/movie_lines.txt') as f:
        for raw in f:
            parts = raw.decode('iso-8859-1', 'ignore').rstrip('\n').split(SEP)
            if len(parts) == 5:
                lines[parts[0]] = parts[4]

    pairs = []
    with z.open('cornell movie-dialogs corpus/movie_conversations.txt') as f:
        for raw in f:
            parts = raw.decode('iso-8859-1', 'ignore').rstrip('\n').split(SEP)
            if len(parts) != 4:
                continue
            ids = re.findall(r'L\d+', parts[3])
            for a, b in zip(ids, ids[1:]):
                if a in lines and b in lines:
                    pairs.append((lines[a], lines[b]))

    def clean(s):
        s = ' '.join(s.split())                      # collapse whitespace
        ok = 0 < len(s) <= MAX_LINE and all(32 <= ord(c) < 127 for c in s)
        return s if ok else None                     # printable ASCII only

    ok = []
    for a, b in pairs:
        a, b = clean(a), clean(b)
        if a and b:
            ok.append((a, b))
    random.Random(0).shuffle(ok)
    ok = ok[:MAX_PAIRS]

    with open('chat.txt', 'w', encoding='utf-8') as f:
        for a, b in ok:
            f.write(f"User: {a}\nBot: {b}\n\n")
    print(f"{len(ok):,} exchanges -> chat.txt ({os.path.getsize('chat.txt') / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
