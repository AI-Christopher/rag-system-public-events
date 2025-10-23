import pytest
import time
from src.core.embedding import get_embedding_model, get_embed_texts
from langchain_mistralai import MistralAIEmbeddings

# --- Test pour get_embedding_model ---

def test_get_embedding_model_success(mocker):
    """
    Vérifie que la fonction initialise et retourne un objet MistralAIEmbeddings
    quand la clé API est présente.
    """
    # 1. On simule les dépendances externes
    # On simule la fonction load_dotenv pour qu'elle ne fasse rien
    mocker.patch('src.core.embedding.load_dotenv')
    # On simule os.getenv pour qu'il retourne une fausse clé
    mocker.patch('os.getenv', return_value='fake_api_key_123')
    # On simule la classe MistralAIEmbeddings
    mock_class = mocker.patch('src.core.embedding.MistralAIEmbeddings')
    
    # 2. On exécute la fonction
    model = get_embedding_model()
    
    # Est-ce que la classe a été initialisée avec les bons arguments ?
    mock_class.assert_called_once_with(
        mistral_api_key='fake_api_key_123',
        model='mistral-embed'
    )
    # Est-ce que la fonction a retourné l'objet (simulé) ?
    assert model is not None

def test_get_embedding_model_failure_no_key(mocker):
    """
    Vérifie que la fonction lève bien une ValueError si la clé API est absente.
    """
    # 1. On simule os.getenv pour qu'il retourne None
    mocker.patch('src.core.embedding.load_dotenv')
    mocker.patch('os.getenv', return_value=None)
    
    # 2. On exécute et on vérifie que l'erreur est levée
    with pytest.raises(ValueError, match="La clé API MISTRAL_API_KEY n'est pas définie"):
        get_embedding_model()


# --- Test pour get_embed_texts ---

def test_get_embed_texts_multiple_batches(mocker, capsys):
    """
    Vérifie que la fonction traite les textes par lots, gère les pauses
    et les exceptions, et retourne la liste complète des vecteurs.
    """
    # 1. On prépare nos fausses données d'entrée
    # On crée 51 textes pour forcer 2 lots (le batch_size est 50)
    texts_to_embed = ["texte"] * 51 
    
    # 2. On simule les dépendances
    
    # On crée un faux objet 'embedding_model'
    mock_model = mocker.Mock(spec=MistralAIEmbeddings)
    # On définit ce qu'il doit retourner à chaque appel (side_effect)
    mock_model.embed_documents.side_effect = [
        [[0.1] * 50], # 1er appel (lot de 50) : échoue
        Exception("Erreur API 429"),
        [[0.2]], # 2e appel (lot de 1) : réussit
    ]

    # On simule 'time.sleep' pour qu'il ne fasse rien
    mock_sleep = mocker.patch('time.sleep')
    
    # On simule 'tqdm' pour qu'il retourne simplement l'itérateur
    mocker.patch('src.core.embedding.tqdm', lambda x, **kwargs: x)
    
    # 3. On exécute la fonction
    # Note : on simule 3 passages dans la boucle : 
    # Batch 1 (0-50) -> Succès
    # Batch 2 (50-100) -> Échec
    # Batch 3 (100-150) -> Succès
    # Pour cela, on modifie les side_effect et les textes
    
    # --- Re-préparation pour un test plus complet ---
    texts_to_embed = ["text"] * 101 # 3 lots: 50, 50, 1
    
    mock_model.embed_documents.side_effect = [
        [[0.1, 0.2]] * 50,         # Lot 1 (50 textes) -> Succès
        Exception("Erreur API"), # Lot 2 (50 textes) -> Échec
        [[0.3, 0.4]] * 1           # Lot 3 (1 texte)  -> Succès
    ]
    
    # 4. Exécution
    vectors = get_embed_texts(texts_to_embed, mock_model)
    
    # 5. Vérification
    
    # On doit avoir 51 vecteurs (50 du lot 1 + 1 du lot 3)
    assert len(vectors) == 51
    assert vectors[0] == [0.1, 0.2] # Vérifie le premier vecteur
    assert vectors[50] == [0.3, 0.4] # Vérifie le dernier vecteur
    
    # On vérifie que 'embed_documents' a été appelé 3 fois
    assert mock_model.embed_documents.call_count == 3
    
    # On vérifie que 'time.sleep' a été appelé 2 fois (pour les 2 lots réussis)
    assert mock_sleep.call_count == 2
    
    # On vérifie que le message d'erreur a bien été affiché
    captured = capsys.readouterr()
    assert "Une erreur est survenue sur le lot 50-100: Erreur API" in captured.out

def test_get_embed_texts_empty_list(mocker):
    """Vérifie le comportement si la liste d'entrée est vide."""
    
    mock_model = mocker.Mock()
    mock_sleep = mocker.patch('time.sleep')
    
    vectors = get_embed_texts([], mock_model)
    
    assert len(vectors) == 0
    assert mock_model.embed_documents.call_count == 0
    assert mock_sleep.call_count == 0