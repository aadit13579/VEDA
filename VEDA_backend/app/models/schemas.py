from pydantic import BaseModel, Field
from typing import List, Optional

class Region(BaseModel):
    bbox: List[float] = Field(..., min_items=4, max_items=4) # [x1, y1, x2, y2]
    type: str
    reading_order: Optional[int] = None

class PageData(BaseModel):
    regions: List[Region]
    page_layout: Optional[str] = None

class LayoutPayload(BaseModel):
    layout_data: List[PageData]