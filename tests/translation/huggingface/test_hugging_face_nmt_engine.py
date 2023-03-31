from machine.translation.huggingface import HuggingFaceNmtEngine


def test_strings() -> None:
    engine = HuggingFaceNmtEngine("franjamonga/translate", batch_size=8, device=0)
    results = engine.translate_n_batch(
        n=2,
        segments=[
            "Me llamo Wolfgang y vivo en Berlin",
            "Los ingredientes de una tortilla de patatas son: huevos, patatas y cebolla",
        ],
    )
    assert results[0][0].translation == "My name is Wolfgang and I live in Berlin"


def test_tokens() -> None:
    engine = HuggingFaceNmtEngine("franjamonga/translate", batch_size=8, device=0)
    results = engine.translate_n_batch(
        n=2,
        segments=[
            ["▁Me", "▁llamo", "▁Wolf", "gan", "g", "▁y", "▁vivo", "▁en", "▁Berlin"],
            [
                "▁Los",
                "▁ingredientes",
                "▁de",
                "▁una",
                "▁tortilla",
                "▁de",
                "▁patatas",
                "▁son",
                ":",
                "▁huevos",
                ",",
                "▁patatas",
                "▁y",
                "▁cebolla",
            ],
        ],
    )
    assert results[0][0].translation == "My name is Wolfgang and I live in Berlin"
