import machine.webapi.clearml_nmt_engine_build_job as cml

args = {
    "engine_id": "645d317a6d7b74cbf8d54239",
    "build_id": "645d317b6d7b74cbf8d54244",
    "src_lang": "es",
    "trg_lang": "en",
    "max_step": 1000,
}
cml.run(args)
