from .data_loader import fetch_events
from .processing import list_to_df, clean_df, filter_and_dedup, create_chunks_with_metadata
from .embedding import get_embedding_model, get_embed_texts
from .faiss_manager import create_faiss_index_from_vectors

def run_indexing_pipeline(region: str = "Occitanie"):
    """Exécute le pipeline complet de création de l'index FAISS."""

    print("--- Lancement du pipeline d'indexation ---")

    # 1. Initialiser le modèle
    embedding_model = get_embedding_model()

    # 2. Récupérer et préparer les données
    list_events = fetch_events(region=region)
    if not list_events:
        print("Aucun événement récupéré. Arrêt du pipeline.")
        return False

    print(f"-> {len(list_events)} événements récupérés.")
    df = list_to_df(list_events)
    df_cleaned = clean_df(df)
    df_final = filter_and_dedup(df_cleaned)
    chunks, metadatas = create_chunks_with_metadata(df_final)

    # 3. Générer les embeddings
    vectors = get_embed_texts(chunks, embedding_model)

    # 4. Créer et sauvegarder l'index
    if vectors and len(vectors) == len(chunks):
        create_faiss_index_from_vectors(chunks, vectors, metadatas, embedding_model)
        print("Pipeline d'indexation terminé avec succès.")
        return True
    else:
        print("Erreur: Le nombre de vecteurs ne correspond pas aux chunks.")
        return False