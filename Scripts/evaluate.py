import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Importer les fonctions de votre chatbot
from chatbot import create_rag_chain, get_retriever
from embedding import get_embedding_model
from langchain_mistralai.chat_models import ChatMistralAI

def run_ragas_evaluation():
    """Prépare les données et lance l'évaluation avec Ragas."""
    
    # --- 1. Préparation de la chaîne RAG ---
    embedding_model = get_embedding_model()
    retriever = get_retriever(embedding_model)
    from chatbot import create_prompt_template # Assurez-vous que cette fonction est importable
    prompt = create_prompt_template()
    rag_chain = create_rag_chain(retriever, prompt, embedding_model)

    # --- 2. Création du jeu de données de test ---
    # Pour Ragas, nous avons besoin de 'question' et 'ground_truth' (réponse de référence)
    eval_questions = [
        "Je cherche un atelier créatif pour les enfants à Toulouse",
        "Y a-t-il des expositions d'art en Occitanie ?",
        "Est ce qu'il y a eu des visites de cave à vin à Vézénobres ?",
        "Quel événement de noël est prévu à Montpellier en décembre 2025 ?"
    ]
    eval_ground_truths = [
        "L'Atelier créatif et l'Atelier Les petits créateurs sont deux options à Toulouse pour les enfants.",
        "Oui, il y a plusieurs expositions, notamment sur des thèmes comme l'art contemporain.",
        "Oui, il y a eu des visites de cave à vin à Vézénobres dans le cadre des Journées du Patrimoine",
        "Je n'ai pas trouvé d'information spécifique relative à un événement de noël à Montpellier pour la période de décembre 2025."
    ]

    # --- 3. Générer les réponses et récupérer le contexte pour chaque question ---
    answers = []
    contexts = []
    for question in eval_questions:
        # On invoque la chaîne pour obtenir la réponse ET le contexte
        response = rag_chain.with_config({"run_name": "eval"}).invoke(
            question, 
        )
        # Pour récupérer le contexte, il faut légèrement modifier la chaîne RAG
        # (voir note ci-dessous) - pour l'instant, on simule
        retrieved_docs = retriever.invoke(question)
        
        answers.append(response)
        contexts.append([doc.page_content for doc in retrieved_docs])

    # --- 4. Formater les données pour Ragas ---
    response_dataset = Dataset.from_dict({
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": eval_ground_truths
    })
    
    # --- 5. Lancer l'évaluation ---
    print("--- Lancement de l'évaluation avec Ragas ---")
    
    # Charger la clé API pour le modèle "juge"
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("La clé API MISTRAL_API_KEY est nécessaire pour le juge.")

    # On définit explicitement le modèle Mistral comme juge
    mistral_judge = ChatMistralAI(model="mistral-small-latest", temperature=0, api_key=api_key)
    mistral_embeddings = get_embedding_model()
    
    result = evaluate(
        dataset=response_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=mistral_judge,
        embeddings=mistral_embeddings
    )

    print("--- Résultats de l'évaluation Ragas ---")
    print(result)

    # Afficher les résultats sous forme de tableau
    df_results = result.to_pandas()
    print(df_results)


if __name__ == '__main__':
    run_ragas_evaluation()