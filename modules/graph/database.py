"""
MongoDB access for knowledge graph generation.

This module centralizes persistence for seeds, generated graphs, checkpoints,
QA pairs, statistics, and validity checks.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import certifi
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient

from modules.graph.knowledge_graph.cokg import CompositionalKnowledgeGraph


CheckpointGraph = Tuple[
    CompositionalKnowledgeGraph,
    List[str],
    Set[str],
    Dict[str, Dict[str, Any]],
    str,
    Dict[str, Any],
]


class DatabaseManager:
    """Database manager for knowledge graph persistence."""

    DATABASE_NAME = "cokg"
    GEN_GRAPHS_COLLECTION = "gen_graphs"
    SEEDS_COLLECTION = "seeds"
    AGENT_STATS_COLLECTION = "agent_stats"
    QA_PAIRS_COLLECTION = "qa_pairs"
    VALIDITY_CHECKS_COLLECTION = "validity_checks"

    def __init__(self) -> None:
        self.mongodb_url = os.getenv("MONGODB_URL")
        if not self.mongodb_url:
            raise ValueError("MONGODB_URL not found in environment variables")

        self._client: Optional[MongoClient] = None

    def _get_client(self) -> MongoClient:
        """Return a cached MongoDB client."""
        if self._client is None:
            self._client = MongoClient(self.mongodb_url, tlsCAFile=certifi.where())
        return self._client

    def _get_database(self) -> Any:
        """Return the configured MongoDB database."""
        return self._get_client()[self.DATABASE_NAME]

    def _collection(self, name: str) -> Any:
        """Return a collection from the configured database."""
        return self._get_database()[name]

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def load_checkpoint_graph(
        self,
        run_id: str,
        graph_id: str,
        graph_type: str = "original",
    ) -> CheckpointGraph:
        """
        Load a checkpoint graph from the generated graphs collection.

        Returns:
            Tuple of ``(cokg, queue, processed_entities, entity_parent_context,
            seed_object_id, stats)``.
        """
        try:
            collection = self._collection(self.GEN_GRAPHS_COLLECTION)
            checkpoint = collection.find_one(
                {"run_id": run_id, "graph_id": graph_id, "type": graph_type}
            )

            if not checkpoint:
                raise ValueError(
                    f"No checkpoint found for run_id: {run_id}, graph_id: {graph_id}"
                )

            print(f"Loading checkpoint: Run {run_id}, Graph {graph_id}")
            print(f"Checkpoint timestamp: {time.ctime(checkpoint.get('timestamp', 0))}")
            print(f"Graph complexity: {checkpoint.get('complexity', 0)} entities")

            cokg = self._deserialize_graph(checkpoint)
            queue = checkpoint.get("queue", [])
            processed_entities = set(cokg.entities) - set(queue)
            entity_parent_context = self._build_entity_parent_context(cokg)
            seed_object_id = checkpoint.get("seed_object_id", "")
            stats = checkpoint.get("stats", {})

            print(f"Queue contains {len(queue)} entities to process")
            print(f"Processed entities: {len(processed_entities)}")
            print(
                "Entity parent context computed for "
                f"{len(entity_parent_context)} entities"
            )
            print(
                "Successfully loaded checkpoint with "
                f"{len(cokg.entities)} entities and {len(cokg.predicates)} predicates"
            )

            if not seed_object_id:
                print("Warning: No seed_object_id found in checkpoint")

            return (
                cokg,
                queue,
                processed_entities,
                entity_parent_context,
                seed_object_id,
                stats,
            )
        except Exception as exc:
            print(f"Error loading checkpoint graph: {exc}")
            raise

    async def load_seed(self, seed_object_id: str) -> Dict[str, Any]:
        """Load a seed entity by MongoDB ObjectId."""
        try:
            collection = self._collection(self.SEEDS_COLLECTION)
            seed_doc = collection.find_one({"_id": ObjectId(seed_object_id)})

            if not seed_doc:
                raise ValueError(
                    f"Seed with ObjectId {seed_object_id} not found in "
                    f"MongoDB collection '{self.SEEDS_COLLECTION}'"
                )

            entity_data = seed_doc.get("entity", {})
            if "stats" in seed_doc:
                entity_data["stats"] = seed_doc["stats"]

            print(
                "Loaded specific seed: "
                f"{entity_data.get('name', 'Unknown')} "
                f"(Category: {entity_data.get('category', 'Unknown')})"
            )
            return entity_data
        except Exception as exc:
            print(f"Error loading seed from MongoDB: {exc}")
            raise

    async def save_seed(
        self,
        entity_data: Dict[str, Any],
        run_id: Optional[str] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> ObjectId:
        """Save a seed entity."""
        try:
            collection = self._collection(self.SEEDS_COLLECTION)
            seed_doc = {
                "entity": entity_data,
                "timestamp": time.time(),
                "stats": stats or {},
            }

            if run_id:
                seed_doc["run_id"] = run_id

            result = collection.insert_one(seed_doc)
            return result.inserted_id
        except Exception as exc:
            print(f"Error saving seed to MongoDB: {exc}")
            raise

    async def get_unfinished_graphs_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Return the latest unfinished original graph checkpoint for each graph."""
        try:
            collection = self._collection(self.GEN_GRAPHS_COLLECTION)
            query = {
                "run_id": run_id,
                "type": "original",
                "queue": {"$exists": True, "$not": {"$size": 0}},
            }
            pipeline = [
                {"$match": query},
                {"$sort": {"timestamp": -1}},
                {"$group": {"_id": "$graph_id", "doc": {"$first": "$$ROOT"}}},
                {"$replaceRoot": {"newRoot": "$doc"}},
            ]

            return list(collection.aggregate(pipeline))
        except Exception as exc:
            print(f"Error getting unfinished graphs for run {run_id}: {exc}")
            raise

    async def get_unused_seeds_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Return seeds from a run that have not been expanded into graphs."""
        try:
            seeds_collection = self._collection(self.SEEDS_COLLECTION)
            graphs_collection = self._collection(self.GEN_GRAPHS_COLLECTION)
            used_seed_ids = graphs_collection.distinct(
                "seed_object_id", {"run_id": run_id}
            )
            used_seed_object_ids = self._valid_object_ids(used_seed_ids)
            query = {"run_id": run_id, "_id": {"$nin": used_seed_object_ids}}

            return list(seeds_collection.find(query).sort("timestamp", 1))
        except Exception as exc:
            print(f"Error getting unused seeds for run {run_id}: {exc}")
            raise

    async def get_unfinished_run_ids(self) -> List[str]:
        """Return run IDs that have unfinished graph queues."""
        try:
            collection = self._collection(self.GEN_GRAPHS_COLLECTION)
            query = {"queue": {"$exists": True, "$not": {"$size": 0}}}
            runs = collection.find(query).sort("timestamp", -1)

            run_ids = []
            seen = set()
            for run in runs:
                run_id = run.get("run_id")
                if run_id and run_id not in seen:
                    run_ids.append(run_id)
                    seen.add(run_id)

            return run_ids
        except Exception as exc:
            print(f"Error getting unfinished run IDs: {exc}")
            raise

    async def check_fuzzy_graph_exists(
        self,
        seed_object_id: str,
        run_id: Optional[str] = None,
    ) -> bool:
        """Return whether a fuzzy graph exists for the seed."""
        return await self._graph_exists(seed_object_id, "fuzzy", run_id)

    async def check_pre_fuzzy_graph_exists(
        self,
        seed_object_id: str,
        run_id: Optional[str] = None,
    ) -> bool:
        """Return whether an original graph exists for the seed."""
        return await self._graph_exists(seed_object_id, "original", run_id)

    async def get_graph_docs_by_run(
        self,
        run_id: str,
        types: Optional[List[str]] = None,
        run_id_prefix: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch generated graph documents by run and graph type."""
        try:
            collection = self._collection(self.GEN_GRAPHS_COLLECTION)
            graph_types = types or ["original", "fuzzy"]
            run_id_query: Any = (
                {"$regex": "^" + re.escape(run_id)} if run_id_prefix else run_id
            )
            query = {"run_id": run_id_query, "type": {"$in": graph_types}}

            return list(collection.find(query))
        except Exception as exc:
            print(f"Error fetching graph docs by run: {exc}")
            raise

    async def save_generated_graph(
        self,
        cokg: CompositionalKnowledgeGraph,
        run_id: str,
        graph_id: str,
        seed_object_id: str,
        new_entities: Optional[List[str]] = None,
        graph_type: str = "original",
        stats: Optional[Dict[str, Any]] = None,
    ) -> ObjectId:
        """Save or update a generated knowledge graph."""
        try:
            collection = self._collection(self.GEN_GRAPHS_COLLECTION)
            graph_doc = self._serialize_graph(
                cokg=cokg,
                run_id=run_id,
                graph_id=graph_id,
                seed_object_id=seed_object_id,
                new_entities=new_entities,
                graph_type=graph_type,
                stats=stats,
            )

            selector = {"run_id": run_id, "graph_id": graph_id, "type": graph_type}
            result = collection.update_one(selector, {"$set": graph_doc}, upsert=True)
            doc_id = (
                result.upserted_id or collection.find_one(selector, {"_id": 1})["_id"]
            )

            print(
                f"Saved/Updated graph {graph_id} ({graph_type}) in MongoDB. "
                f"Document ID: {doc_id}"
            )
            return doc_id
        except Exception as exc:
            print(f"Error saving generated graph to MongoDB: {exc}")
            raise

    async def save_consolidated_stats(
        self,
        run_id: str,
        graph_id: str,
        seed_object_id: str,
        stats: Dict[str, Any],
    ) -> ObjectId:
        """Save consolidated statistics for a graph generation run."""
        try:
            collection = self._collection(self.AGENT_STATS_COLLECTION)
            doc = {
                "run_id": run_id,
                "graph_id": graph_id,
                "seed_object_id": seed_object_id,
                "timestamp": time.time(),
                "stats": stats,
            }

            result = collection.insert_one(doc)
            print(
                f"Saved consolidated stats for Graph {graph_id} in MongoDB. "
                f"Document ID: {result.inserted_id}"
            )
            return result.inserted_id
        except Exception as exc:
            print(f"Error saving consolidated stats to MongoDB: {exc}")
            raise

    async def save_qa_pair(
        self,
        run_id: str,
        graph_id: str,
        question: str,
        answer: str,
        stats: Optional[Dict[str, Any]] = None,
    ) -> ObjectId:
        """Save a question-answer pair for a generated graph."""
        try:
            collection = self._collection(self.QA_PAIRS_COLLECTION)
            doc = {
                "run_id": run_id,
                "graph_id": graph_id,
                "question": question,
                "answer": answer,
                "timestamp": time.time(),
                "stats": stats or {},
            }

            result = collection.insert_one(doc)
            print(
                f"Saved QA pair for Graph {graph_id} in MongoDB. "
                f"Document ID: {result.inserted_id}"
            )
            return result.inserted_id
        except Exception as exc:
            print(f"Error saving QA pair to MongoDB: {exc}")
            raise

    async def get_qa_pairs(
        self,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch QA pairs, optionally filtered by run prefix."""
        try:
            collection = self._collection(self.QA_PAIRS_COLLECTION)
            query = {}
            if run_id:
                query["run_id"] = {"$regex": f"^{re.escape(run_id)}"}

            cursor = collection.find(query)
            if limit:
                cursor = cursor.limit(limit)

            qa_pairs = list(cursor)
            for pair in qa_pairs:
                pair["id"] = str(pair.pop("_id"))

            return qa_pairs
        except Exception as exc:
            print(f"Error fetching QA pairs: {exc}")
            raise

    async def get_fuzzy_graphs_by_run_ids(
        self,
        run_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch fuzzy graph documents for the given run IDs."""
        try:
            collection = self._collection(self.GEN_GRAPHS_COLLECTION)
            query = {"type": "fuzzy"}
            if run_ids:
                query["run_id"] = {"$in": run_ids}

            graphs = list(collection.find(query))
            print(f"Fetched {len(graphs)} fuzzy graphs from MongoDB")
            return graphs
        except Exception as exc:
            print(f"Error fetching fuzzy graphs by run_ids: {exc}")
            raise

    async def check_validity_check_exists(self, run_id: str, graph_id: str) -> bool:
        """Return whether a validity check exists for a graph."""
        try:
            collection = self._collection(self.VALIDITY_CHECKS_COLLECTION)
            result = collection.find_one({"run_id": run_id, "graph_id": graph_id})
            return result is not None
        except Exception as exc:
            print(f"Error checking for validity check: {exc}")
            return False

    async def save_validity_check_result(
        self,
        run_id: str,
        graph_id: str,
        results: Dict[str, Any],
        stats: Optional[Dict[str, Any]] = None,
    ) -> ObjectId:
        """Save graph validity check results."""
        try:
            collection = self._collection(self.VALIDITY_CHECKS_COLLECTION)
            doc = {
                "run_id": run_id,
                "graph_id": graph_id,
                "results": results,
                "timestamp": time.time(),
                "stats": stats or {},
            }

            result = collection.insert_one(doc)
            print(
                f"Saved validity check result for Graph {graph_id} in MongoDB. "
                f"Document ID: {result.inserted_id}"
            )
            return result.inserted_id
        except Exception as exc:
            print(f"Error saving validity check result: {exc}")
            raise

    async def _graph_exists(
        self,
        seed_object_id: str,
        graph_type: str,
        run_id: Optional[str] = None,
    ) -> bool:
        """Return whether a generated graph document exists."""
        try:
            collection = self._collection(self.GEN_GRAPHS_COLLECTION)
            query = {"seed_object_id": seed_object_id, "type": graph_type}
            if run_id:
                query["run_id"] = run_id

            return collection.find_one(query) is not None
        except Exception as exc:
            print(f"Error checking for {graph_type} graph: {exc}")
            return False

    def _serialize_graph(
        self,
        cokg: CompositionalKnowledgeGraph,
        run_id: str,
        graph_id: str,
        seed_object_id: str,
        new_entities: Optional[List[str]],
        graph_type: str,
        stats: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Convert a knowledge graph into a MongoDB document."""
        return {
            "run_id": run_id,
            "graph_id": graph_id,
            "seed_object_id": seed_object_id,
            "type": graph_type,
            "timestamp": time.time(),
            "complexity": len(cokg.entities),
            "entities": [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "description": entity.description,
                    "category": entity.category,
                    "is_variable": entity.is_variable,
                    "links": entity.links,
                }
                for entity in cokg.entities.values()
            ],
            "predicates": [
                {
                    "id": predicate.id,
                    "content": predicate.content,
                    "input": predicate.input,
                    "outputs": predicate.outputs,
                    "links": predicate.links,
                }
                for predicate in cokg.predicates.values()
            ],
            "queue": new_entities or [],
            "stats": stats or {},
        }

    def _deserialize_graph(
        self, graph_doc: Dict[str, Any]
    ) -> CompositionalKnowledgeGraph:
        """Build a knowledge graph from a stored graph document."""
        cokg = CompositionalKnowledgeGraph()

        for entity_data in graph_doc.get("entities", []):
            cokg.add_entity(
                id=entity_data["id"],
                name=entity_data["name"],
                description=entity_data["description"],
                category=entity_data.get("category", "Unknown"),
                is_variable=entity_data.get("is_variable", True),
                links=entity_data.get("links", []),
            )

        for predicate_data in graph_doc.get("predicates", []):
            cokg.add_predicate(
                id=predicate_data.get("id"),
                content=predicate_data["content"],
                input=predicate_data["input"],
                outputs=predicate_data["outputs"],
                links=predicate_data.get("links", []),
            )

        return cokg

    def _build_entity_parent_context(
        self,
        cokg: CompositionalKnowledgeGraph,
    ) -> Dict[str, Dict[str, Any]]:
        """Build parent predicate context for each entity in a graph."""
        entity_parent_context: Dict[str, Dict[str, Any]] = {}

        for entity_id in cokg.entities:
            parent_predicates = []
            parent_entity = None

            for predicate in cokg.predicates.values():
                if entity_id not in predicate.outputs:
                    continue

                parent_predicates.append(
                    {
                        "content": predicate.content,
                        "input": predicate.input,
                        "outputs": predicate.outputs,
                        "links": predicate.links,
                    }
                )

                if parent_entity is None:
                    parent_entity = predicate.input

            if parent_predicates:
                entity_parent_context[entity_id] = {
                    "parent_predicates": parent_predicates,
                    "parent_entity": parent_entity,
                }

        return entity_parent_context

    def _valid_object_ids(self, ids: List[Any]) -> List[ObjectId]:
        """Convert valid ObjectId strings and ignore invalid values."""
        object_ids = []
        for value in ids:
            if not value:
                continue

            try:
                object_ids.append(ObjectId(value))
            except (InvalidId, TypeError):
                continue

        return object_ids
