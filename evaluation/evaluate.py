"""Dataset-level evaluation for the final MAPS inference path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from evaluation.run_inference import (
    ADD_TEACHER_MISSING,
    AUX_THR_GRID if False else None,
)
