from typing import cast

from mockito import ANY, mock, when
from mongomock.mongo_client import MongoClient

from machine.tokenization import WhitespaceDetokenizer, WhitespaceTokenizer
from machine.translation import Trainer, TranslationEngine, TranslationModel
from machine.translation.trainer import TrainStats
from machine.webapi import DataFileService, NmtEngineBuildJob, NmtModelFactory, Repository
from machine.webapi.models import BUILD_STATE_PENDING, ENGINE_TYPE_NMT, Build, Engine, Translation


def test_run() -> None:
    env = _TestEnvironment()
    env.job.run(str(env.engine_id), str(env.build_id))


class _TestEnvironment:
    def __init__(self) -> None:
        client = MongoClient()
        self.engines: Repository[Engine] = Repository(client.machine.engines)
        self.engine_id = self.engines.insert(
            Engine(
                sourceLanguageTag="es",
                targetLanguageTag="en",
                type=ENGINE_TYPE_NMT,
                owner="app1",
                isBuilding=False,
                modelRevision=0,
                confidence=0,
                trainedSegmentCount=0,
            )
        )
        self.builds: Repository[Build] = Repository(client.machine.builds)
        self.build_id = self.builds.insert(
            Build(engineRef=self.engine_id, jobId="job1", step=0, state=BUILD_STATE_PENDING)
        )
        self.translations: Repository[Translation] = Repository(client.machine.translations)

        self.data_file_service = cast(DataFileService, mock(DataFileService))
        when(self.data_file_service).create_text_corpora(ANY, ANY).thenReturn({})
        when(self.data_file_service).get_translate_text_ids(ANY).thenReturn({})

        self.source_tokenizer_trainer = cast(Trainer, mock(Trainer))
        when(self.source_tokenizer_trainer).train().thenReturn()
        when(self.source_tokenizer_trainer).save().thenReturn()

        self.target_tokenizer_trainer = cast(Trainer, mock(Trainer))
        when(self.target_tokenizer_trainer).train().thenReturn()
        when(self.target_tokenizer_trainer).save().thenReturn()

        self.model_trainer = cast(Trainer, mock(Trainer))
        when(self.model_trainer).train(ANY, ANY).thenReturn()
        when(self.model_trainer).save().thenReturn()
        stats = TrainStats()
        stats.metrics["bleu"] = 0.0
        setattr(self.model_trainer, "stats", stats)

        self.engine = cast(TranslationEngine, mock(TranslationEngine))
        when(self.engine).__enter__().thenReturn(self.engine)
        when(self.engine).__exit__(ANY, ANY, ANY).thenReturn()

        self.model = cast(TranslationModel, mock(TranslationModel))
        when(self.model).create_engine().thenReturn(self.engine)
        when(self.model).__enter__().thenReturn(self.model)
        when(self.model).__exit__(ANY, ANY, ANY).thenReturn()

        self.nmt_model_factory = cast(NmtModelFactory, mock(NmtModelFactory))
        when(self.nmt_model_factory).init(ANY).thenReturn()
        when(self.nmt_model_factory).create_source_tokenizer_trainer(ANY, ANY).thenReturn(self.source_tokenizer_trainer)
        when(self.nmt_model_factory).create_target_tokenizer_trainer(ANY, ANY).thenReturn(self.target_tokenizer_trainer)
        when(self.nmt_model_factory).create_model_trainer(ANY, ANY).thenReturn(self.model_trainer)
        when(self.nmt_model_factory).create_source_tokenizer(ANY).thenReturn(WhitespaceTokenizer())
        when(self.nmt_model_factory).create_target_detokenizer(ANY).thenReturn(WhitespaceDetokenizer())
        when(self.nmt_model_factory).create_model(ANY).thenReturn(self.model)
        when(self.nmt_model_factory).cleanup(ANY).thenReturn()

        self.job = NmtEngineBuildJob(
            self.engines, self.builds, self.translations, self.data_file_service, self.nmt_model_factory
        )
