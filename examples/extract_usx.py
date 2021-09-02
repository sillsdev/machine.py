import sys

from machine.corpora import UsxFileTextCorpus
from machine.tokenization import NullTokenizer


def main() -> None:
    tokenizer = NullTokenizer()
    corpus = UsxFileTextCorpus(tokenizer, sys.argv[1])

    with open(sys.argv[2], "w", encoding="utf-8") as file:
        for text_segment in corpus.get_segments():
            if text_segment.is_empty:
                continue
            verse_ref = str(text_segment.segment_ref)
            verse_text = "".join(text_segment.segment)
            file.write(f"{verse_ref}\t{verse_text}\n")


if __name__ == "__main__":
    main()
