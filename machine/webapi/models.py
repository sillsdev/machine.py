import sys
from datetime import datetime
from typing import List

from bson.objectid import ObjectId

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


DATA_TYPE_TEXT_CORPUS = "TextCorpus"

CORPUS_TYPE_SOURCE = "Source"
CORPUS_TYPE_TARGET = "Target"
CORPUS_TYPE_BOTH = "Both"

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


class ChangeEvent(TypedDict):
    entityRef: ObjectId
    changeType: str
    revision: int


class Entity(TypedDict, total=False):
    _id: ObjectId
    revision: int


class Build(Entity, total=False):
    engineRef: ObjectId
    jobId: str
    step: int
    percentCompleted: float
    message: str
    state: str
    dateFinished: datetime


class DataFile(Entity, total=False):
    engineRef: ObjectId
    dataType: str
    name: str
    format: str
    corpusType: str
    corpusId: str
    filename: str


class Engine(Entity, total=False):
    sourceLanguageTag: str
    targetLanguageTag: str
    engineType: str
    owner: str
    isBuilding: bool
    modelRevision: int
    confidence: float
    trainedSegmentCount: int


class Translation(Entity):
    engineRef: ObjectId
    corpusId: str
    textId: str
    refs: List[str]
    text: str
