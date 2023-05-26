from machine.jobs.build_nmt_engine import run

args = {
    "model_type": "huggingface",
    "engine_id": "646e09a1470dd37b0511845f",
    "build_id": "646e09a1470dd37b0511845f",
    "src_lang": "spa_Latn",
    "trg_lang": "eng_Latn",
    "max_steps": 10,
    "clearml": True,
}
run(args)
