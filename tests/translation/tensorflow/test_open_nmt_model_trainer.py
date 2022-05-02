from pathlib import Path

from machine.corpora import TextFileTextCorpus
from machine.tokenization import WhitespaceTokenizer
from machine.translation.tensorflow import OpenNmtModel, OpenNmtModelTrainer


def test_train() -> None:
    model_dir = Path("C:\\develop\\temp\\train_model")
    src_corpus = TextFileTextCorpus(model_dir / "corpus" / "src.txt")
    trg_corpus = TextFileTextCorpus(model_dir / "corpus" / "trg.txt")
    parallel_corpus = src_corpus.align_rows(trg_corpus).tokenize(WhitespaceTokenizer())

    config = {
        "auto_config": True,
        "model_dir": str(model_dir),
        "data": {
            "source_vocabulary": "onmt.vocab",
            "target_vocabulary": "onmt.vocab",
            "train_features_file": "train.src.txt",
            "train_labels_file": "train.trg.txt",
            "eval_features_file": "val.src.txt",
            "eval_labels_file": "val.trg.txt",
        },
        "train": {"max_step": 100, "average_last_checkpoints": 0},
    }
    parent_config = {
        "auto_config": True,
        "model_dir": "C:\\develop\\data\\MT\\experiments\\de-to-en-WMT2020+Bibles\\run",
        "data": {
            "source_vocabulary": "C:\\develop\\data\\MT\\experiments\\de-to-en-WMT2020+Bibles\\src-onmt.vocab",
            "target_vocabulary": "C:\\develop\\data\\MT\\experiments\\de-to-en-WMT2020+Bibles\\trg-onmt.vocab",
        },
    }
    trainer = OpenNmtModelTrainer("TransformerBase", config, parallel_corpus, parent_config, mixed_precision=True)
    trainer.train()
    trainer.save()


def test_model_train() -> None:
    model_dir = Path("C:\\develop\\temp\\train_model")
    src_corpus = TextFileTextCorpus(model_dir / "corpus" / "src.txt")
    trg_corpus = TextFileTextCorpus(model_dir / "corpus" / "trg.txt")
    parallel_corpus = src_corpus.align_rows(trg_corpus).tokenize(WhitespaceTokenizer())

    config = {
        "auto_config": True,
        "model_dir": str(model_dir),
        "data": {
            "source_vocabulary": "onmt.vocab",
            "target_vocabulary": "onmt.vocab",
            "train_features_file": "train.src.txt",
            "train_labels_file": "train.trg.txt",
            "eval_features_file": "val.src.txt",
            "eval_labels_file": "val.trg.txt",
        },
        "train": {"max_step": 100, "average_last_checkpoints": 0},
    }
    parent_config = {
        "auto_config": True,
        "model_dir": "C:\\develop\\data\\MT\\experiments\\de-to-en-WMT2020+Bibles\\run",
        "data": {
            "source_vocabulary": "C:\\develop\\data\\MT\\experiments\\de-to-en-WMT2020+Bibles\\src-onmt.vocab",
            "target_vocabulary": "C:\\develop\\data\\MT\\experiments\\de-to-en-WMT2020+Bibles\\trg-onmt.vocab",
        },
    }
    model = OpenNmtModel("TransformerBase", config, parent_config, mixed_precision=True)
    trainer = model.create_trainer(parallel_corpus)
    trainer.train()
    trainer.save()
