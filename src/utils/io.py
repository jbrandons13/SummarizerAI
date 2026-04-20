import json
from pathlib import Path
from typing import Type, TypeVar, Any
from pydantic import BaseModel
from src.exceptions import IOError as vsIOError
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def load_json_as_model(file_path: Path, model_class: Type[T]) -> T:
    """
    Load a JSON file and validate it against a Pydantic model.
    
    Args:
        file_path: Path to the JSON file.
        model_class: The Pydantic model class to validate against.
        
    Returns:
        An instance of the Pydantic model.
    """
    if not file_path.exists():
        raise vsIOError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return model_class.model_validate(data)
    except Exception as e:
        raise vsIOError(f"Failed to load or validate JSON from {file_path}: {e}")

def save_model_as_json(model_instance: BaseModel, file_path: Path, indent: int = 4):
    """
    Save a Pydantic model instance as a JSON file.
    
    Args:
        model_instance: The Pydantic model instance to save.
        file_path: Path to the output JSON file.
        indent: Indentation level for the JSON.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(model_instance.model_dump_json(indent=indent))
    except Exception as e:
        raise vsIOError(f"Failed to save model to {file_path}: {e}")

def load_raw_json(file_path: Path) -> Any:
    """
    Load a raw JSON file without Pydantic validation.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        The content of the JSON file.
    """
    if not file_path.exists():
        raise vsIOError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise vsIOError(f"Failed to load JSON from {file_path}: {e}")
