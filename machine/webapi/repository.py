from typing import Any, Generic, List, Mapping, MutableMapping, Optional, Sequence, TypeVar, Union, cast

from bson.objectid import ObjectId
from pymongo.collection import Collection, ReturnDocument
from pymongo.cursor import Cursor
from pymongo.database import Database
from pymongo.write_concern import WriteConcern

from .models import ENTITY_CHANGE_DELETE, ENTITY_CHANGE_INSERT, ENTITY_CHANGE_UPDATE, ChangeEvent, Entity

TEntity = TypeVar("TEntity", bound=Entity)


class Repository(Generic[TEntity]):
    def __init__(self, collection: Collection[TEntity], is_subscribable: bool = False) -> None:
        self._collection = collection
        self._is_subscribable = is_subscribable
        if is_subscribable:
            database = cast(Database, collection.database)
            self._change_events: Collection[ChangeEvent] = database.get_collection(
                collection.name + "_log"
            ).with_options(write_concern=WriteConcern(w=0))

    def get(self, filter: Union[Mapping[str, Any], ObjectId, str]) -> Optional[TEntity]:
        if isinstance(filter, ObjectId):
            filter = {"_id": filter}
        elif isinstance(filter, str):
            filter = {"_id": ObjectId(filter)}
        return self._collection.find_one(filter)

    def get_all(self, filter: Mapping[str, Any]) -> Cursor[TEntity]:
        return self._collection.find(filter)

    def exists(self, filter: Mapping[str, Any]) -> bool:
        return self._collection.count_documents(filter) > 0

    def insert_many(self, entities: Sequence[TEntity]) -> None:
        for entity in entities:
            entity["revision"] = 1
        res = self._collection.insert_many(cast(List[MutableMapping[str, Any]], entities))
        if res.acknowledged and self._is_subscribable:
            for entity in entities:
                self._change_events.insert_many(
                    {
                        "entityRef": entity.get("_id"),
                        "changeType": ENTITY_CHANGE_INSERT,
                        "revision": entity.get("revision"),
                    }
                    for entity in entities
                )

    def insert(self, entity: TEntity) -> ObjectId:
        entity["revision"] = 1
        res = self._collection.insert_one(cast(MutableMapping[str, Any], entity))
        if res.acknowledged and self._is_subscribable:
            self._change_events.insert_one(
                {"entityRef": entity.get("_id"), "changeType": ENTITY_CHANGE_INSERT, "revision": entity.get("revision")}
            )
        return res.inserted_id

    def update(
        self, filter: Union[Mapping[str, Any], ObjectId, str], update: Mapping[str, Any], upsert: bool = False
    ) -> Optional[TEntity]:
        if isinstance(filter, ObjectId):
            filter = {"_id": filter}
        elif isinstance(filter, str):
            filter = {"_id": ObjectId(filter)}
        update = dict(update)
        if "$inc" in update:
            update["$inc"]["revision"] = 1
        else:
            update["$inc"] = {"revision": 1}
        entity = self._collection.find_one_and_update(
            filter, update, return_document=ReturnDocument.AFTER, upsert=upsert
        )
        if entity is not None and self._is_subscribable:
            self._change_events.insert_one(
                {"entityRef": entity.get("_id"), "changeType": ENTITY_CHANGE_UPDATE, "revision": entity.get("revision")}
            )
        return entity

    def delete(self, filter: Mapping[str, Any]) -> Optional[TEntity]:
        entity = self._collection.find_one_and_delete(filter)
        if entity is not None and self._is_subscribable:
            self._change_events.insert_one(
                {
                    "entityRef": entity.get("_id"),
                    "changeType": ENTITY_CHANGE_DELETE,
                    "revision": entity.get("revision", 1) + 1,
                }
            )
        return entity
