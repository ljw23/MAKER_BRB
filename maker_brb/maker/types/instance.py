from typing import Dict, List, Set
from pydantic import BaseModel
from copy import deepcopy
from pprint import pprint


class Feature_value(BaseModel):
    feature_id: int
    value: float


class Instance(BaseModel):
    id: str = ''
    feature_values: List[Feature_value]
