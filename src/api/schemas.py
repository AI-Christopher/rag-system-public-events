from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Modèle de la requête pour /ask"""
    question: str = Field(
        ...,  # Signifie que le champ est requis
        description="La question en langage naturel à poser au RAG.",
        example="Y a-t-il des expositions d'art en Occitanie ?"
    )

class QueryResponse(BaseModel):
    """Modèle de la réponse pour /ask"""
    answer: str = Field(
        ...,
        description="La réponse générée par le système RAG.",
        example="Oui, il y a plusieurs expositions d'art..."
    )

class RebuildResponse(BaseModel):
    """Modèle de la réponse pour /rebuild"""
    status: str = Field(
        ...,
        description="Le statut de la reconstruction de l'index.",
        example="ok"
    )
    message: str = Field(
        ...,
        description="Un message détaillant le résultat de l'opération.",
        example="La reconstruction de l'index a été lancée en arrière-plan."
    )