from fastapi import FastAPI, HTTPException, BackgroundTasks
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


app = FastAPI(
    title="API pour le RAG d'événements",
    description="Permet de poser des questions et de gérer l'index vectoriel.",
    version="1.0.0"
)

@app.post("/ask", response_model=QueryResponse)
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


@app.post("/rebuild", response_model=RebuildResponse)
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