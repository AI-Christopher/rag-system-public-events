from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="La question en langage naturel à poser au RAG.",
        # Change 'example=' par 'json_schema_extra='
        json_schema_extra={"example": "Y a-t-il des expositions d'art en Occitanie ?"}
    )

class QueryResponse(BaseModel):
    answer: str = Field(
        ...,
        description="La réponse générée par le système RAG.",
        json_schema_extra={"example": "Oui, il y a plusieurs expositions d'art..."}
    )

class RebuildResponse(BaseModel):
    status: str = Field(
        ..., 
        json_schema_extra={"example": "ok"}
    )
    message: str = Field(
        ..., 
        json_schema_extra={"example": "La reconstruction de l'index a été lancée..."}
    )