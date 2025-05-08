"""
Entity Type Definitions and Hierarchies.

This module defines the base entity type system, type hierarchy management,
type attribute definitions, and domain-specific type extensions.
"""

import logging
import json
from typing import Dict, List, Optional, Union, Any, Set, Tuple
import os
import uuid
from dataclasses import dataclass, field

from db.models.entity_models import EntityType as DbEntityType

logger = logging.getLogger(__name__)


@dataclass
class EntityAttribute:
    """Definition of an attribute for an entity type."""
    name: str
    description: str = ""
    data_type: str = "string"  # string, number, boolean, date, etc.
    required: bool = False
    multi_valued: bool = False
    enum_values: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type,
            "required": self.required,
            "multi_valued": self.multi_valued,
            "enum_values": self.enum_values
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityAttribute':
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            data_type=data.get("data_type", "string"),
            required=data.get("required", False),
            multi_valued=data.get("multi_valued", False),
            enum_values=data.get("enum_values", [])
        )


class EntityType:
    """
    Entity type definition with hierarchy and attributes.
    
    Represents a type of entity with hierarchical relationship to other types
    and a set of attributes that define the structure of entities of this type.
    """
    
    def __init__(self, name: str, description: str = "", parent: Optional['EntityType'] = None,
                 attributes: Optional[List[EntityAttribute]] = None, type_id: Optional[str] = None,
                 domain: str = "generic"):
        """
        Initialize a new entity type.
        
        Args:
            name: Name of the entity type
            description: Description of the entity type
            parent: Optional parent entity type
            attributes: List of attributes for this entity type
            type_id: Optional type ID (generated if not provided)
            domain: Domain this entity type belongs to
        """
        self.id = type_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.parent = parent
        self.children = []
        self.attributes = attributes or []
        self.domain = domain
        
        # If we have a parent, add this as a child to the parent
        if self.parent:
            self.parent.add_child(self)
    
    def add_child(self, child: 'EntityType') -> None:
        """
        Add a child entity type.
        
        Args:
            child: Child entity type to add
        """
        if child not in self.children:
            self.children.append(child)
            child.parent = self
    
    def add_attribute(self, attribute: EntityAttribute) -> None:
        """
        Add an attribute to this entity type.
        
        Args:
            attribute: Attribute to add
        """
        # Check if attribute already exists
        existing = next((a for a in self.attributes if a.name == attribute.name), None)
        if existing:
            # Update existing attribute
            idx = self.attributes.index(existing)
            self.attributes[idx] = attribute
        else:
            # Add new attribute
            self.attributes.append(attribute)
    
    def get_full_name(self) -> str:
        """
        Get the full hierarchical name of this entity type.
        
        Returns:
            Full name with parent types (e.g., "PERSON.PLAYER.GOALKEEPER")
        """
        if not self.parent:
            return self.name
        
        return f"{self.parent.get_full_name()}.{self.name}"
    
    def is_subtype_of(self, potential_parent: Union[str, 'EntityType']) -> bool:
        """
        Check if this type is a subtype of the potential parent.
        
        Args:
            potential_parent: Entity type or type name to check against
            
        Returns:
            True if this is a subtype of the potential parent, False otherwise
        """
        # Convert string to type name for comparison
        parent_name = potential_parent if isinstance(potential_parent, str) else potential_parent.get_full_name()
        
        # Check if this type's full name starts with the parent name
        return self.get_full_name().startswith(parent_name)
    
    def get_all_attributes(self, include_parent_attributes: bool = True) -> List[EntityAttribute]:
        """
        Get all attributes for this entity type, optionally including parent attributes.
        
        Args:
            include_parent_attributes: Whether to include attributes from parent types
            
        Returns:
            List of attributes
        """
        if not include_parent_attributes or not self.parent:
            return self.attributes.copy()
        
        # Get parent attributes
        parent_attributes = self.parent.get_all_attributes()
        
        # Combine with this type's attributes, giving precedence to this type's attributes
        result = parent_attributes.copy()
        
        # Add or override with this type's attributes
        for attr in self.attributes:
            # Check if attribute already exists in result
            existing = next((a for a in result if a.name == attr.name), None)
            if existing:
                # Replace existing attribute
                idx = result.index(existing)
                result[idx] = attr
            else:
                # Add new attribute
                result.append(attr)
        
        return result
    
    def to_dict(self, include_children: bool = False) -> Dict[str, Any]:
        """
        Convert entity type to dictionary representation.
        
        Args:
            include_children: Whether to include children recursively
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": self.id,
            "name": self.name,
            "full_name": self.get_full_name(),
            "description": self.description,
            "domain": self.domain,
            "attributes": [attr.to_dict() for attr in self.attributes]
        }
        
        if include_children and self.children:
            result["children"] = [child.to_dict(include_children=True) for child in self.children]
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional['EntityType'] = None) -> 'EntityType':
        """
        Create entity type from dictionary representation.
        
        Args:
            data: Dictionary representation
            parent: Optional parent entity type
            
        Returns:
            Created entity type
        """
        # Create attributes
        attributes = [
            EntityAttribute.from_dict(attr_data) for attr_data in data.get("attributes", [])
        ]
        
        # Create this type
        entity_type = cls(
            name=data["name"],
            description=data.get("description", ""),
            parent=parent,
            attributes=attributes,
            type_id=data.get("id"),
            domain=data.get("domain", "generic")
        )
        
        # Create children if present
        for child_data in data.get("children", []):
            EntityType.from_dict(child_data, parent=entity_type)
        
        return entity_type
    
    def save_to_database(self) -> DbEntityType:
        """
        Save entity type to the database.
        
        Returns:
            Database EntityType object
        """
        # Create or get database entity type
        db_entity_type = DbEntityType.get_by_id(self.id) if self.id else None
        
        if not db_entity_type:
            db_entity_type = DbEntityType(
                name=self.name,
                description=self.description,
                parent_id=self.parent.id if self.parent else None,
                domain=self.domain
            )
            db_entity_type.save()
        else:
            # Update existing
            db_entity_type.name = self.name
            db_entity_type.description = self.description
            db_entity_type.parent_id = self.parent.id if self.parent else None
            db_entity_type.domain = self.domain
            db_entity_type.save()
        
        # Save attributes
        db_entity_type.set_attributes([attr.to_dict() for attr in self.attributes])
        
        # Update ID if newly created
        if not self.id:
            self.id = db_entity_type.id
        
        # Recursively save children
        for child in self.children:
            child.save_to_database()
        
        return db_entity_type


class EntityTypeRegistry:
    """
    Registry for entity types with lookup capabilities.
    
    This class manages a collection of entity types, allowing lookup by name,
    ID, or hierarchical full name.
    """
    
    def __init__(self, domain: str = "generic"):
        """
        Initialize a new entity type registry.
        
        Args:
            domain: Domain this registry is for
        """
        self.domain = domain
        self.root_types = []
        self._type_cache = {}  # Map of ID to EntityType
        self._name_cache = {}  # Map of full name to EntityType
    
    def add_root_type(self, entity_type: EntityType) -> None:
        """
        Add a root entity type to the registry.
        
        Args:
            entity_type: Root entity type to add
        """
        self.root_types.append(entity_type)
        self._update_caches(entity_type)
    
    def _update_caches(self, entity_type: EntityType) -> None:
        """
        Update the type cache with an entity type and its children.
        
        Args:
            entity_type: Entity type to add to cache
        """
        self._type_cache[entity_type.id] = entity_type
        self._name_cache[entity_type.get_full_name()] = entity_type
        
        for child in entity_type.children:
            self._update_caches(child)
    
    def get_type_by_id(self, type_id: str) -> Optional[EntityType]:
        """
        Get entity type by ID.
        
        Args:
            type_id: ID to look up
            
        Returns:
            Entity type if found, None otherwise
        """
        return self._type_cache.get(type_id)
    
    def get_type_by_name(self, name: str) -> Optional[EntityType]:
        """
        Get entity type by full name.
        
        Args:
            name: Full name to look up (e.g., "PERSON.PLAYER")
            
        Returns:
            Entity type if found, None otherwise
        """
        return self._name_cache.get(name)
    
    def get_subtypes(self, parent_type: Union[str, EntityType]) -> List[EntityType]:
        """
        Get all subtypes of a given entity type.
        
        Args:
            parent_type: Parent entity type or type name
            
        Returns:
            List of subtypes
        """
        # Convert string to type
        if isinstance(parent_type, str):
            parent_type = self.get_type_by_name(parent_type)
            if not parent_type:
                logger.warning(f"Entity type not found: {parent_type}")
                return []
        
        # Get all types that are subtypes of the parent
        return [t for t in self._type_cache.values() if t.is_subtype_of(parent_type)]
    
    def get_all_types(self) -> List[EntityType]:
        """
        Get all entity types in the registry.
        
        Returns:
            List of all entity types
        """
        return list(self._type_cache.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert registry to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "domain": self.domain,
            "types": [t.to_dict(include_children=True) for t in self.root_types]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityTypeRegistry':
        """
        Create registry from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created registry
        """
        registry = cls(domain=data.get("domain", "generic"))
        
        for type_data in data.get("types", []):
            root_type = EntityType.from_dict(type_data)
            registry.add_root_type(root_type)
        
        return registry
    
    def save_to_database(self) -> List[DbEntityType]:
        """
        Save all entity types to the database.
        
        Returns:
            List of database EntityType objects
        """
        db_types = []
        
        for root_type in self.root_types:
            db_type = root_type.save_to_database()
            db_types.append(db_type)
        
        return db_types
    
    @classmethod
    def load_from_database(cls, domain: str) -> 'EntityTypeRegistry':
        """
        Load entity types from the database.
        
        Args:
            domain: Domain to load types for
            
        Returns:
            Loaded registry
        """
        registry = cls(domain=domain)
        
        # Get root types from database
        db_root_types = DbEntityType.get_root_types(domain=domain)
        
        # Create registry
        for db_root_type in db_root_types:
            root_type = cls._load_entity_type_from_db(db_root_type)
            registry.add_root_type(root_type)
        
        return registry
    
    @staticmethod
    def _load_entity_type_from_db(db_entity_type: DbEntityType, parent: Optional[EntityType] = None) -> EntityType:
        """
        Load entity type from database.
        
        Args:
            db_entity_type: Database entity type
            parent: Optional parent entity type
            
        Returns:
            Loaded entity type
        """
        # Create attributes
        attributes = []
        for attr_data in db_entity_type.get_attributes():
            attributes.append(EntityAttribute.from_dict(attr_data))
        
        # Create entity type
        entity_type = EntityType(
            name=db_entity_type.name,
            description=db_entity_type.description,
            parent=parent,
            attributes=attributes,
            type_id=db_entity_type.id,
            domain=db_entity_type.domain
        )
        
        # Add children
        for db_child in db_entity_type.get_children():
            child = EntityTypeRegistry._load_entity_type_from_db(db_child, parent=entity_type)
            entity_type.add_child(child)
        
        return entity_type


def create_standard_entity_types() -> EntityTypeRegistry:
    """
    Create standard entity types for general use.
    
    Returns:
        Registry with standard entity types
    """
    registry = EntityTypeRegistry(domain="standard")
    
    # Create PERSON type
    person = EntityType("PERSON", "A human individual", domain="standard")
    person.add_attribute(EntityAttribute("name", "Full name of the person"))
    person.add_attribute(EntityAttribute("gender", "Gender of the person", enum_values=["Male", "Female", "Other"]))
    person.add_attribute(EntityAttribute("nationality", "Nationality of the person"))
    person.add_attribute(EntityAttribute("birth_date", "Birth date of the person", data_type="date"))
    registry.add_root_type(person)
    
    # Create ORGANIZATION type
    org = EntityType("ORGANIZATION", "An organization or institution", domain="standard")
    org.add_attribute(EntityAttribute("name", "Name of the organization"))
    org.add_attribute(EntityAttribute("type", "Type of organization", 
                                     enum_values=["Company", "Government", "NGO", "Educational", "Sports", "Other"]))
    org.add_attribute(EntityAttribute("location", "Primary location of the organization"))
    org.add_attribute(EntityAttribute("founded", "Foundation date", data_type="date"))
    registry.add_root_type(org)
    
    # Create LOCATION type
    location = EntityType("LOCATION", "A physical location", domain="standard")
    location.add_attribute(EntityAttribute("name", "Name of the location"))
    location.add_attribute(EntityAttribute("type", "Type of location",
                                          enum_values=["Country", "City", "Address", "Landmark", "Region", "Other"]))
    location.add_attribute(EntityAttribute("coordinates", "Geographic coordinates"))
    registry.add_root_type(location)
    
    # Create EVENT type
    event = EntityType("EVENT", "A noteworthy occurrence", domain="standard")
    event.add_attribute(EntityAttribute("name", "Name of the event"))
    event.add_attribute(EntityAttribute("start_date", "Start date of the event", data_type="date"))
    event.add_attribute(EntityAttribute("end_date", "End date of the event", data_type="date"))
    event.add_attribute(EntityAttribute("location", "Location of the event"))
    registry.add_root_type(event)
    
    # Create PRODUCT type
    product = EntityType("PRODUCT", "A commercial product", domain="standard")
    product.add_attribute(EntityAttribute("name", "Name of the product"))
    product.add_attribute(EntityAttribute("manufacturer", "Manufacturer of the product"))
    product.add_attribute(EntityAttribute("category", "Category of the product"))
    product.add_attribute(EntityAttribute("release_date", "Release date of the product", data_type="date"))
    registry.add_root_type(product)
    
    # Create DATE type
    date = EntityType("DATE", "A reference to a specific date or time", domain="standard")
    date.add_attribute(EntityAttribute("value", "ISO format date value", data_type="date"))
    date.add_attribute(EntityAttribute("type", "Type of date reference",
                                      enum_values=["Absolute", "Relative", "Range", "Recurring"]))
    registry.add_root_type(date)
    
    # Create QUANTITY type
    quantity = EntityType("QUANTITY", "A measurement with a unit", domain="standard")
    quantity.add_attribute(EntityAttribute("value", "Numeric value", data_type="number"))
    quantity.add_attribute(EntityAttribute("unit", "Unit of measurement"))
    quantity.add_attribute(EntityAttribute("type", "Type of quantity",
                                         enum_values=["Money", "Distance", "Weight", "Duration", "Other"]))
    registry.add_root_type(quantity)
    
    return registry


def merge_entity_type_registries(base_registry: EntityTypeRegistry, 
                                extension_registry: EntityTypeRegistry) -> EntityTypeRegistry:
    """
    Merge two entity type registries, combining their type hierarchies.
    
    Args:
        base_registry: Base registry to extend
        extension_registry: Registry with extensions to add
        
    Returns:
        New combined registry
    """
    # Create new registry with base domain
    merged_registry = EntityTypeRegistry(domain=base_registry.domain)
    
    # Copy types from base registry
    for root_type in base_registry.root_types:
        merged_registry.add_root_type(root_type)
    
    # Merge in types from extension registry
    for ext_root_type in extension_registry.root_types:
        # Check if this type exists in base registry
        base_type = merged_registry.get_type_by_name(ext_root_type.name)
        
        if base_type:
            # Merge type into existing type
            _merge_entity_types(base_type, ext_root_type)
        else:
            # Add as new root type
            merged_registry.add_root_type(ext_root_type)
    
    return merged_registry


def _merge_entity_types(base_type: EntityType, extension_type: EntityType) -> None:
    """
    Merge extension type into base type.
    
    Args:
        base_type: Base type to extend
        extension_type: Type with extensions to add
    """
    # Merge attributes
    for attr in extension_type.attributes:
        base_type.add_attribute(attr)
    
    # Merge children
    for ext_child in extension_type.children:
        # Check if this child exists in base type
        base_child = next((c for c in base_type.children if c.name == ext_child.name), None)
        
        if base_child:
            # Recursively merge
            _merge_entity_types(base_child, ext_child)
        else:
            # Add as new child
            ext_child.parent = base_type
            base_type.add_child(ext_child)
