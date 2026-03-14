from pydantic import BaseModel
from typing import List

class RegionModel(BaseModel):
    bbox: List[float] # Must be length 4
    type: str

class PageModel(BaseModel):
    regions: List[RegionModel]

class LayoutResponse(BaseModel):
    layout_data: List[PageModel]