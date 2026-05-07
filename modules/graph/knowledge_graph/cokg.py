from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Entity:
    id: str  # e.g., "E0", "E1", or "X", "Y"
    name: str  # The real name of the entity
    description: str  # A fuzzy description of the entity
    category: str = "Unknown"
    is_variable: bool = True
    links: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Entity to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "is_variable": self.is_variable,
            "links": self.links
        }

    def __repr__(self):
        prefix = "V@" if self.is_variable else "C@"
        links_str = f", links={self.links}" if self.links else ""
        return f"{prefix}{self.id}({self.name}: {self.description}, cat={self.category}{links_str})"

@dataclass
class Predicate:
    id: str  # e.g., "P0", "P1"
    content: str       # e.g., "<|-E0-|> won an award of <|-E1-|> in a year of 1980"
    input: str         # Entity ID (the described entity, Ein)
    outputs: List[str] # Entity IDs (the describing entities, Eouts)
    links: List[str] = field(default_factory=list)

    def __repr__(self):
        links_str = f" [links: {', '.join(self.links)}]" if self.links else ""
        return f"{self.id}:{self.input} -> {self.content}{links_str}"

class CompositionalKnowledgeGraph:
    """
    Compositional Knowledge Graph (CoKG) abstraction.
    A directed acyclic graph G_C = (V_E ∪ V_P, E),
    where V_E is the set of entities and V_P is the set of predicates.
    Each predicate connects one described entity (input) to multiple describing entities (outputs).
    """
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.predicates: Dict[str, Predicate] = {}

    def add_entity(self, name: str, description: str, category: str = "Unknown", is_variable: bool = True, id: Optional[str] = None, links: Optional[List[str]] = None) -> Entity:
        """Adds an entity to the graph."""
        if id is None:
            id = f"E{len(self.entities)}"
        
        if id in self.entities:
            raise ValueError(f"Entity ID '{id}' already exists in CoKG.")
            
        entity = Entity(id=id, name=name, description=description, category=category, is_variable=is_variable, links=links or [])
        self.entities[id] = entity
        return entity

    def add_predicate(self, content: str, input: str, outputs: List[str], id: Optional[str] = None, links: Optional[List[str]] = None) -> Predicate:
        """Adds a predicate to the graph, connecting a described entity to describing entities."""
        if id is None:
            id = f"P{len(self.predicates)}"
        
        # Validate that all referenced entities exist
        if input not in self.entities:
            raise ValueError(f"Input entity '{input}' not found in CoKG. Please add it first.")
        for out in outputs:
            if out not in self.entities:
                raise ValueError(f"Output entity '{out}' not found in CoKG. Please add it first.")
            
        predicate = Predicate(id=id, content=content, input=input, outputs=outputs, links=links or [])
        self.predicates[id] = predicate
        return predicate

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the CoKG to a dictionary."""
        return {
            "entities": [
                {
                    "id": e.id, 
                    "name": e.name, 
                    "description": e.description, 
                    "category": e.category, 
                    "is_variable": e.is_variable,
                    "links": e.links
                }
                for e in self.entities.values()
            ],
            "predicates": [
                {
                    "id": p.id, 
                    "content": p.content, 
                    "input": p.input, 
                    "outputs": p.outputs,
                    "links": p.links
                }
                for p in self.predicates.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompositionalKnowledgeGraph':
        """Creates a CoKG from a dictionary representation."""
        cokg = cls()
        # Entities first to satisfy predicate dependencies
        for e_data in data.get("entities", []):
            cokg.add_entity(
                id=e_data["id"],
                name=e_data["name"],
                description=e_data["description"],
                category=e_data.get("category", "Unknown"),
                is_variable=e_data.get("is_variable", True),
                links=e_data.get("links", [])
            )
        for p_data in data.get("predicates", []):
            cokg.add_predicate(
                id=p_data.get("id"),
                content=p_data["content"],
                input=p_data["input"],
                outputs=p_data["outputs"],
                links=p_data.get("links", [])
            )
        return cokg

    def __repr__(self):
        entities_str = "\n  ".join([repr(e) for e in self.entities.values()])
        predicates_str = "\n  ".join([repr(p) for p in self.predicates.values()])
        return f"CoKG(\n  Entities:\n  {entities_str}\n  Predicates:\n  {predicates_str}\n)"

    def fix_entity_types(self):
        """
        Fixes the is_variable flag for all entities:
        - Set to True (variable) if the entity is an input to at least one predicate.
        - Set to False (constant) otherwise.
        """
        input_entities = {p.input for p in self.predicates.values()}
        for entity_id, entity in self.entities.items():
            entity.is_variable = entity_id in input_entities
