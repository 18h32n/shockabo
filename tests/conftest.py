import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_project_dir(request):
    """Create a temporary project directory structure."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create directory structure
    directories = [
        'src/infrastructure',
        'configs',
        'scripts/platform_deploy',
        'data',
        'output',
        'cache',
        'logs',
        'tests'
    ]

    for dir_path in directories:
        (temp_dir / dir_path).mkdir(parents=True, exist_ok=True)

    def cleanup():
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    request.addfinalizer(cleanup)
    return temp_dir