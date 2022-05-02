from machine.translation.tensorflow import OpenNmtModel


def test_translate() -> None:
    config = {
        "auto_config": True,
        "model_dir": "C:/develop/temp/test_model",
        "data": {
            "source_vocabulary": "onmt.vocab",
            "target_vocabulary": "onmt.vocab",
        },
    }
    with OpenNmtModel("TransformerBase", config, mixed_precision=True) as model, model.create_engine() as engine:
        result = engine.translate(
            ["▁malasing", "▁ma", "▁aubina", "▁di", "▁vatawagalaai", "▁la", "lozang", "▁iesu", "."]
        )
        assert result.target_segment == ["▁and", "▁so", "▁all", "▁the", "▁people", "▁were", "▁following", "▁jesus", "."]
