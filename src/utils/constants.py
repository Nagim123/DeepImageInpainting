"""
File containing all constants including paths.
"""

import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()

PATH_TO_MODELS = os.path.join(script_path, f"../models")