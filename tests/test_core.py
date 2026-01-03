import pytest
import os
import sys

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.marketplace import AVAILABLE_MODELS

def test_marketplace_structure():
    """Verify all models have required fields."""
    required = ["id", "name", "url_model", "filename_model"]
    for model in AVAILABLE_MODELS:
        for field in required:
            assert field in model, f"Model {model.get('id')} missing {field}"

def test_model_filenames():
    """Verify filenames are safe strings."""
    for model in AVAILABLE_MODELS:
        assert ".." not in model["filename_model"]
        assert "/" not in model["filename_model"]
