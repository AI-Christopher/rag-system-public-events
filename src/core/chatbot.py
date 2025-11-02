from .embedding import get_embedding_model
from .faiss_manager import load_faiss_index
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_retriever(embedding_model, index_path="data/faiss_index"):
    """
    Crée et retourne un retriever à partir d'un index FAISS existant.
    """
    # Charger la base de données vectorielle
    vectorstore = load_faiss_index(embedding_model, index_path)
    
    # Transformer la base de données en un "retriever"
    # search_kwargs={'k': 3} signifie qu'on récupérera les 3 chunks les plus pertinents.
    return vectorstore.as_retriever(search_kwargs={'k': 5})


def create_prompt_template():
    """
    Crée et retourne un template de prompt pour le chatbot RAG.
    """
    template = """
    Tu es un assistant spécialisé dans la recommandation d'événements publics.
    Réponds à la question de l'utilisateur en te basant uniquement sur le contexte suivant.
    Sois aimable, concis et présente les informations de manière claire, par exemple avec des listes à puces.
    Si le contexte ne contient pas la réponse, dis simplement que tu n'as pas trouvé d'information à ce sujet.

    Contexte :
    {context}

    Question :
    {question}

    Réponse :
    """
    return ChatPromptTemplate.from_template(template)


def create_rag_chain(retriever, prompt, embedding_model):
    """
    Crée et retourne une chaîne RAG complète.
    """
    # Initialiser le modèle de chat Mistral
    llm = ChatMistralAI(
        model="open-mistral-7b",
        temperature=0.1, # Peu de créativité pour s'en tenir aux faits
        api_key=embedding_model.mistral_api_key # On réutilise la clé
    )
    
    # Fonction pour formater les documents récupérés
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Création de la chaîne RAG avec la syntaxe LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


if __name__ == '__main__':
    # 1. Initialiser le modèle d'embedding (nécessaire pour le retriever)
    embedding_model = get_embedding_model()
    
    # 2. Créer le retriever
    retriever = get_retriever(embedding_model)
    
    # 3. Créer le prompt
    prompt = create_prompt_template()
    
    # 4. Créer la chaîne RAG
    rag_chain = create_rag_chain(retriever, prompt, embedding_model)
    
    print("🤖 Chatbot d'événements prêt ! Posez vos questions (tapez 'exit' pour quitter).")
    
    while True:
        query = input("Vous > ")
        if query.lower() == 'exit':
            break
            
        # 5. Invoquer la chaîne pour obtenir une réponse
        response = rag_chain.invoke(query)
        
        print(f"Bot > {response}")