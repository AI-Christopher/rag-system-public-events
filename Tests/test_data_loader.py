import pytest
import requests
from src.core.data_loader import fetch_events # On importe la fonction à tester

def test_fetch_events_success_with_pagination(mocker):
    """
    Teste le cas idéal : l'API répond, et la pagination fonctionne sur 2 pages.
    """
    
    # 1. On prépare nos fausses réponses d'API
    
    # Réponse pour la première page (offset=0)
    mock_response_page1 = mocker.Mock()
    mock_response_page1.raise_for_status.return_value = None
    mock_response_page1.json.return_value = {
        'total_count': 3, # Il y a 3 événements au total
        'results': [
            {'id': 'evt1', 'titre': 'Concert Rock'},
            {'id': 'evt2', 'titre': 'Expo Photo'}
        ]
    }
    
    # Réponse pour la deuxième page (offset=100)
    mock_response_page2 = mocker.Mock()
    mock_response_page2.raise_for_status.return_value = None
    mock_response_page2.json.return_value = {
        'total_count': 3,
        'results': [
            {'id': 'evt3', 'titre': 'Atelier Cuisine'}
        ]
    }
    
    # 2. On configure le "mock" de requests.get
    # Il retournera page1 au premier appel, puis page2 au second.
    mocker.patch('requests.get', side_effect=[mock_response_page1, mock_response_page2])
    
    # 3. On exécute la fonction
    events = fetch_events(region="Occitanie")
    
    # 4. On vérifie les résultats
    assert len(events) == 3, "Doit récupérer le total de 3 événements."
    assert events[0]['id'] == 'evt1'
    assert events[2]['id'] == 'evt3'

def test_fetch_events_no_events_found(mocker, capsys):
    """
    Teste le cas où l'API répond qu'il n'y a aucun événement.
    """
    # 1. Préparation de la fausse réponse (total_count = 0)
    mock_response_empty = mocker.Mock()
    mock_response_empty.raise_for_status.return_value = None
    mock_response_empty.json.return_value = {'total_count': 0, 'results': []}
    
    # 2. Configuration du mock
    mocker.patch('requests.get', return_value=mock_response_empty)
    
    # 3. Exécution
    events = fetch_events(region="Occitanie")
    
    # 4. Vérification
    assert len(events) == 0, "Doit retourner une liste vide."
    
    # On vérifie aussi que le message correct a été affiché
    captured = capsys.readouterr()
    assert "Aucun événement trouvé." in captured.out

def test_fetch_events_http_error(mocker, capsys):
    """
    Teste la gestion d'une erreur HTTP (ex: 404, 500)
    """
    
    # 1. Créer un faux objet "response" qui contient les attributs 'url' et 'text'
    mock_response = mocker.Mock()
    mock_response.url = "http://fake-url-that-failed.com"
    mock_response.text = '{"error": "Not Found"}' # Le contenu de l'erreur
    
    # 2. Créer l'erreur HTTPError et lui attacher notre fausse réponse
    http_err = requests.exceptions.HTTPError("404 Not Found")
    http_err.response = mock_response # C'est l'étape clé qui manquait
    
    # 3. Configurer requests.get pour qu'il lève cette erreur complète
    mocker.patch('requests.get', side_effect=http_err)
    
    # 4. Exécution
    events = fetch_events(region="Occitanie")
    
    # 5. Vérification
    assert len(events) == 0, "Doit retourner une liste vide en cas d'erreur HTTP."
    captured = capsys.readouterr()
    
    # On vérifie que les deux prints de notre bloc except sont bien passés
    assert "URL de la requête qui a échoué : http://fake-url-that-failed.com" in captured.out
    assert 'Contenu de la réponse : {"error": "Not Found"}' in captured.out


def test_fetch_events_network_error(mocker, capsys):
    """
    Teste la gestion d'une erreur réseau (ex: Timeout)
    """
    # Ce test fonctionne, car l'erreur 'RequestException' n'a pas d'attribut 'response',
    # et notre code n'essaie pas d'y accéder.
    
    mocker.patch('requests.get', side_effect=requests.exceptions.RequestException("Connection Timeout"))
    
    events = fetch_events(region="Occitanie")
    
    assert len(events) == 0
    captured = capsys.readouterr()
    assert "Erreur réseau: Connection Timeout" in captured.out