from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from .schemas import QueryRequest, QueryResponse, RebuildResponse

# Ajoute le chemin racine pour importer 'src.core'
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importe l'instance unique de notre service RAG
# C'est ici que les modèles sont chargés au DÉMARRAGE de l'API
try:
    from src.core.rag_service import rag_service
    print("RAG Service importé avec succès dans l'API.")
except Exception as e:
    print(f"ERREUR CRITIQUE au démarrage : {e}")
    rag_service = None

# Ajout de 'tags_metadata' pour organiser l'API Swagger
tags_metadata = [
    {
        "name": "Système RAG",
        "description": "Points de terminaison pour interroger le RAG.",
    },
    {
        "name": "Administration",
        "description": "Opérations de maintenance de l'index.",
    },
]

app = FastAPI(
    title="API pour le RAG d'événements",
    description="Permet de poser des questions et de gérer l'index vectoriel.",
    version="1.0.0",
    openapi_tags=tags_metadata
)

@app.post(
    "/ask", 
    response_model=QueryResponse,
    tags=["Système RAG"],  # Regroupe cet endpoint
    summary="Interroger le système RAG",
    description=(
        "Pose une question en langage naturel au système RAG. "
        "Le système trouvera les documents pertinents dans la base vectorielle "
        "et utilisera un LLM (MistralAI) pour générer une réponse."
    ),
    responses={
        400: {"description": "La question fournie est vide."},
        503: {"description": "Le service RAG n'a pas pu être initialisé (ex: modèle non trouvé)."}
    }
)
async def ask_question(query: QueryRequest):
    """
    Pose une question au système RAG et obtient une réponse augmentée.
    """
    if not query.question or query.question.strip() == "":
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service RAG non initialisé.")

    answer = rag_service.ask(query.question)
    return QueryResponse(answer=answer)


@app.post(
    "/rebuild", 
    response_model=RebuildResponse,
    tags=["Administration"], # Regroupe cet endpoint
    summary="Lancer la reconstruction de l'index",
    description=(
        "Déclenche une reconstruction complète de l'index vectoriel FAISS. "
        "Il s'agit d'une **opération longue** (plusieurs minutes) qui s'exécute en arrière-plan. "
        "L'API répond immédiatement pour confirmer que la tâche est lancée."
    ),
    # Utilise 202 "Accepté" pour indiquer une tâche de fond
    status_code=status.HTTP_202_ACCEPTED, 
    responses={
        503: {"description": "Le service RAG n'a pas pu être initialisé."}
    }
)
async def rebuild_vector_index(background_tasks: BackgroundTasks):
    """
    Lance la reconstruction complète de l'index vectoriel FAISS.
    Ceci est une opération longue (plusieurs minutes).
    L'API répond immédiatement pendant que la tâche s'exécute en arrière-plan.
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service RAG non initialisé.")

    # Ajoute la tâche de reconstruction à l'arrière-plan
    background_tasks.add_task(rag_service.rebuild_index)

    return RebuildResponse(
        status="ok",
        message="La reconstruction de l'index a été lancée en arrière-plan."
    )