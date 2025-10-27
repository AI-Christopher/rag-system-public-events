from .embedding import get_embedding_model
from .faiss_manager import load_faiss_index
from .chatbot import create_rag_chain, create_prompt_template
from .pipeline import run_indexing_pipeline

class RAGService:
    """
    Service encapsulant la logique RAG pour une utilisation facile par l'API.
    Charge les modèles au démarrage et les garde en mémoire.
    """
    def __init__(self):
        print("Initialisation du RAG Service...")
        self.embedding_model = None
        self.rag_chain = None
        self.load_components()
        print("RAG Service prêt.")

    def load_components(self):
        """Charge l'index FAISS et construit la chaîne RAG."""
        try:
            self.embedding_model = get_embedding_model()
            # 1. Charger le retriever
            vectorstore = load_faiss_index(self.embedding_model)
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            # 2. Créer le prompt
            prompt = create_prompt_template()
            # 3. Créer la chaîne RAG
            self.rag_chain = create_rag_chain(retriever, prompt, self.embedding_model)
            print("Composants RAG chargés avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement des composants RAG : {e}")
            print("Veuillez d'abord construire l'index avec 'build_index.py'.")
            self.rag_chain = None

    def ask(self, question: str) -> str:
        """Pose une question à la chaîne RAG."""
        if not self.rag_chain:
            return "Erreur : Le système RAG n'est pas initialisé. Veuillez d'abord construire l'index."

        print(f"Interrogation de la chaîne RAG avec la question : '{question}'")
        return self.rag_chain.invoke(question)

    def rebuild_index(self):
        """Lance la reconstruction de l'index et recharge les composants."""
        print("Début de la reconstruction de l'index...")
        success = run_indexing_pipeline()
        if success:
            print("Reconstruction terminée. Rechargement des composants...")
            self.load_components()
            return "Index reconstruit et rechargé avec succès."
        else:
            return "Erreur lors de la reconstruction de l'index."

# Créer une instance unique (Singleton) qui sera importée par l'API
# C'est ce qui garantit que les modèles ne sont chargés qu'UNE SEULE FOIS.
rag_service = RAGService()