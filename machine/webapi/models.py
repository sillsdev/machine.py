import sys
from datetime import datetime
from typing import List

from bson.objectid import ObjectId

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import NotRequired, TypedDict


CORPUS_TYPE_TEXT = "Text"

FILE_FORMAT_TEXT = "Text"
FILE_FORMAT_PARATEXT = "Paratext"

BUILD_STATE_PENDING = "Pending"
BUILD_STATE_ACTIVE = "Active"
BUILD_STATE_COMPLETED = "Completed"
BUILD_STATE_FAULTED = "Faulted"
BUILD_STATE_CANCELED = "Canceled"

ENTITY_CHANGE_NONE = "None"
ENTITY_CHANGE_INSERT = "Insert"
ENTITY_CHANGE_UPDATE = "Update"
ENTITY_CHANGE_DELETE = "Delete"

ENGINE_TYPE_SMT_TRANSFER = "SmtTransfer"
ENGINE_TYPE_NMT = "Nmt"


class ChangeEvent(TypedDict):
    entityRef: ObjectId
    changeType: str
    revision: int


class Entity(TypedDict):
    _id: NotRequired[ObjectId]
    revision: NotRequired[int]


class Build(Entity):
    parentRef: ObjectId
    jobId: NotRequired[str]
    step: int
    percentCompleted: NotRequired[float]
    message: NotRequired[str]
    state: str
    dateFinished: NotRequired[datetime]


class DataFile(TypedDict):
    _id: ObjectId
    languageTag: str
    name: str
    filename: str
    textId: NotRequired[str]


class Corpus(Entity):
    owner: str
    name: str
    type: str
    format: str
    files: List[DataFile]


class TranslationEngineCorpus(TypedDict):
    corpusRef: ObjectId
    pretranslate: bool


class TranslationEngine(Entity):
    sourceLanguageTag: str
    targetLanguageTag: str
    type: str
    owner: str
    corpora: List[TranslationEngineCorpus]
    isBuilding: bool
    modelRevision: int
    confidence: float
    trainSize: int


class Pretranslation(Entity):
    translationEngineRef: ObjectId
    corpusRef: ObjectId
    textId: str
    refs: List[str]
    text: str
