import pandas as pd
import pytest
#import re
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.processing import clean_df, filter_and_dedup, create_chunks_with_metadata

@pytest.fixture
def dirty_dataframe_for_cleaning() -> pd.DataFrame:
    """
    Crée un DataFrame de test "sale", conçu pour tester tous les cas de figure
    de la fonction clean_df.
    """
    data = {
        'titre': [
            '  Événement <b>Spécial</b> ♫',  # HTML, espaces, caractère spécial
            'Autre Show',
            None  # Valeur manquante
        ],
        'description': [
            'Courte desc.',
            '<p>Juste du texte</p> →', # HTML, caractère spécial
            ''
        ],
        'description_complete': [
            '   Description longue avec    espaces superflus. ',
            None, # Valeur manquante
            ''
        ],
        'mots_cles': [
            ['Art', 'Musique'], # Une liste
            None, # Valeur manquante
            [] # Une liste vide
        ],
        'lieu': ['Le Grand Lieu', 'Petit Lieu', ''],
        'ville': ['Paris', 'Lyon', None]
    }
    return pd.DataFrame(data)

def test_clean_df_comprehensive(dirty_dataframe_for_cleaning):
    """
    Teste la fonction clean_df pour s'assurer qu'elle nettoie, concatène
    et filtre comme attendu.
    """
    # 1. On exécute la fonction à tester sur nos données sales
    df_resultat = clean_df(dirty_dataframe_for_cleaning)
    
    # --- 2. On vérifie la structure du DataFrame final ---
    
    # La 3ème ligne de nos données de test devrait être supprimée car son 'texte_complet' sera vide
    assert len(df_resultat) == 2, "La fonction devrait supprimer les lignes devenues vides."
    
    # La colonne 'texte_complet' doit avoir été créée
    assert 'texte_complet' in df_resultat.columns
    
    # --- 3. On vérifie le contenu précis, ligne par ligne ---
    
    # Cas de la première ligne (la plus complexe)
    expected_text_1 = "Événement Spécial . Courte desc. . Description longue avec espaces superflus. . Art, Musique . Le Grand Lieu . Paris"
    assert df_resultat.loc[0, 'texte_complet'] == expected_text_1

    # Cas de la deuxième ligne (avec des valeurs manquantes)
    expected_text_2 = "Autre Show . Juste du texte . Petit Lieu . Lyon"
    assert df_resultat.loc[1, 'texte_complet'] == expected_text_2
    
    # --- 4. On vérifie la propreté des colonnes individuelles ---

    # La colonne 'mots_cles' doit maintenant être une chaîne de caractères propre
    assert df_resultat.loc[0, 'mots_cles'] == "Art, Musique"
    assert df_resultat.loc[1, 'mots_cles'] == "" # None est devenu une chaîne vide

    # La colonne 'titre' ne doit plus contenir de HTML ou de caractères spéciaux
    assert df_resultat.loc[0, 'titre'] == "Événement Spécial"


@pytest.fixture
def dataframe_for_filtering() -> pd.DataFrame:
    """
    Crée un DataFrame de test pour la fonction filter_and_dedup.
    Il contient des textes courts, des doublons exacts et des doublons normalisés.
    """
    long_text = "Ceci est un texte suffisamment long pour être conservé. Il est unique." # 70+ chars
    data = {
        'texte_complet': [
            long_text,
            "Texte trop court.", # Cas 1: Doit être supprimé (longueur < 50)
            long_text, # Cas 2: Doublon exact
            "CECI EST UN TEXTE SUFFISAMMENT LONG POUR ÊTRE CONSERVÉ. IL EST UNIQUE.", # Cas 3: Doublon normalisé
            "Un autre texte long et unique qui doit absolument être gardé." # Autre cas valide
        ],
        'titre': ['Titre 1', 'Titre 2', 'Titre 3', 'Titre 4', 'Titre 5']
    }
    return pd.DataFrame(data)


def test_filter_and_dedup(dataframe_for_filtering):
    """
    Vérifie que la fonction supprime bien les textes courts et les doublons.
    """
    # 1. On définit une limite de caractères courte pour le test
    min_chars_test = 50
    
    # 2. On exécute la fonction à tester
    df_resultat = filter_and_dedup(dataframe_for_filtering, min_chars=min_chars_test)
    
    # 3. On vérifie les résultats
    
    # On s'attend à ne garder que 2 lignes :
    # - La première ligne (long_text)
    # - La dernière ligne (texte long et unique)
    assert len(df_resultat) == 2, "La fonction devrait garder 2 lignes uniques et assez longues."
    
    # On vérifie que la colonne temporaire a bien été supprimée
    assert 'empreinte_texte' not in df_resultat.columns
    
    # On vérifie que l'index a bien été réinitialisé
    assert list(df_resultat.index) == [0, 1]
    
    # On vérifie le contenu exact
    assert df_resultat.loc[0, 'titre'] == 'Titre 1'
    assert df_resultat.loc[1, 'titre'] == 'Titre 5'


@pytest.fixture
def dataframe_for_chunking() -> pd.DataFrame:
    """
    Crée un DataFrame de test pour la fonction create_chunks_with_metadata.
    """
    # Texte conçu pour être coupé en 3 par un chunk_size de 20
    long_text_to_split = "Partie un. Partie deux. Partie trois." # 44 chars
    
    data = {
        'id': ['evt_123', 'evt_456'],
        'titre': ['Événement long', 'Événement court'],
        'ville': ['Paris', 'Lyon'],
        'texte_complet': [
            long_text_to_split, # Sera coupé en 3 chunks
            "Texte court."      # Sera 1 seul chunk
        ],
        'colonne_inutile': ['A ignorer', 'B ignorer'] # Doit être filtrée
    }
    return pd.DataFrame(data)


def test_create_chunks_with_metadata(dataframe_for_chunking):
    """
    Vérifie que le texte est bien "chunké" et que chaque chunk
    a les bonnes métadonnées (et seulement celles-ci).
    """
    # 1. On utilise une petite taille de chunk pour forcer la division
    chunk_size_test = 20
    chunk_overlap_test = 5
    
    # 2. On exécute la fonction à tester
    texts, metadatas = create_chunks_with_metadata(
        dataframe_for_chunking, 
        chunk_size=chunk_size_test, 
        chunk_overlap=chunk_overlap_test
    )
    
    # 3. On vérifie les résultats
    
    # On s'attend à 4 chunks au total :
    # "Partie un." (11)
    # "Partie deux." (12)
    # "Partie trois finale." (20)
    # "Texte court." (12)
    # Note: Le split exact dépend de RecursiveCharacterTextSplitter, 
    # mais le nombre total de chunks (3+1) est ce qui nous intéresse.
    assert len(texts) == 4
    assert len(metadatas) == 4
    
    # --- Vérification du premier chunk (de l'événement 1) ---
    assert texts[0] == "Partie un"
    assert metadatas[0]['titre'] == "Événement long"
    assert metadatas[0]['ville'] == "Paris"
    assert metadatas[0]['id'] == "evt_123"
    assert metadatas[0]['chunk_id'] == "evt_123_0" # ID du chunk 0
    assert metadatas[0]['source'] == "openagenda"
    
    # --- Vérification du dernier chunk (de l'événement 2) ---
    assert texts[3] == "Texte court."
    assert metadatas[3]['titre'] == "Événement court"
    assert metadatas[3]['ville'] == "Lyon"
    assert metadatas[3]['id'] == "evt_456"
    assert metadatas[3]['chunk_id'] == "evt_456_0" # ID du chunk 0
    
    # --- Vérification de la structure des métadonnées ---
    # On vérifie qu'une colonne non désirée n'a pas été incluse
    assert 'colonne_inutile' not in metadatas[0]
    assert 'texte_complet' not in metadatas[0]