from fastapi.testclient import TestClient
# import pytest
# import requests
# import time


# 1. Importez votre application FastAPI
try:
    from src.api.main import app
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que PYTHONPATH est bien configuré.")
    # On force un échec si l'import ne marche pas
    app = None 

# 2. Créez un "client" de test
# Ceci remplace le besoin d'avoir un serveur uvicorn qui tourne.
client = TestClient(app)

def test_ask_endpoint():
    """Teste le endpoint /ask"""
    print("--- Test du endpoint /ask ---")
    
    response = client.post(
        "/ask",
        json={"question": "Je cherche un atelier pour les enfants"}
    )
    
    # 4. Les assertions restent les mêmes
    assert response.status_code == 200, f"Erreur de l'API: {response.json()}"
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    # if response.status_code == 200:
    #     print("Requête réussie.")
    #     data = response.json()
    #     print(f"Réponse de l'API : {data['answer']}")
    # else:
    #     print(f"Erreur {response.status_code} : {response.text}")

def test_rebuild_endpoint():
    """Teste le endpoint /rebuild"""
    print("\n--- Test du endpoint /rebuild ---")
    
    response = client.post("/rebuild")
    
    # 4. Les assertions restent les mêmes
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "ok"
    assert "La reconstruction de l'index a été lancée" in data["message"]
    # if response.status_code == 200:
    #     print("Requête réussie.")
    #     data = response.json()
    #     print(f"Réponse de l'API : {data['message']}")
    #     print("L'index est en cours de reconstruction en arrière-plan...")
    # else:
    #     print(f"Erreur {response.status_code} : {response.text}")

def test_ask_endpoint_empty_query():
    """Teste que l'API gère bien une question vide."""
    print("\n--- Test du endpoint /ask (question vide) ---")
    
    response = client.post(
        "/ask",
        json={"question": ""} # Question vide
    )
    
    # L'API doit retourner une erreur 400 (Bad Request)
    assert response.status_code == 400
    assert "La question ne peut pas être vide" in response.json()["detail"]

# if __name__ == "__main__":
#     # Assurez-vous que l'API est lancée (uvicorn src.api.main:app)
    
#     test_ask_endpoint()
    
#     # Décommentez pour tester la reconstruction
#     # test_rebuild_endpoint()
    
#     print("\nTests terminés.")