from pydantic import BaseModel

class QueryRequest(BaseModel):
    """Modèle de la requête pour /ask"""
    question: str

class QueryResponse(BaseModel):
    """Modèle de la réponse pour /ask"""
    answer: str

class RebuildResponse(BaseModel):
    """Modèle de la réponse pour /rebuild"""
    status: str
    message: str