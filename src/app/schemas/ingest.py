from pydantic import BaseModel

class IngestTextRequest(BaseModel):
    title: str
    text: str

