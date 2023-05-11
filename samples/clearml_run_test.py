import machine.webapi.clearml_nmt_engine_build_job as cml

args = {
    "engine_id": "existingengineid",
    "build_id": "existingbuildid",
    "src_lang": "es",
    "trg_lang": "en",
    "max_step": 1000,
    "save_model": False,
}
cml.run(args)
