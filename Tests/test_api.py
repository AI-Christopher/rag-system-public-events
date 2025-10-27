import requests
import time

API_URL = "http://127.0.0.1:8000"

def test_ask_endpoint():
    """Teste le endpoint /ask"""
    print("--- Test du endpoint /ask ---")
    
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": "Je cherche un atelier pour les enfants"}
    )
    
    if response.status_code == 200:
        print("Requête réussie.")
        data = response.json()
        print(f"Réponse de l'API : {data['answer']}")
    else:
        print(f"Erreur {response.status_code} : {response.text}")

def test_rebuild_endpoint():
    """Teste le endpoint /rebuild"""
    print("\n--- Test du endpoint /rebuild ---")
    
    response = requests.post(f"{API_URL}/rebuild")
    
    if response.status_code == 200:
        print("Requête réussie.")
        data = response.json()
        print(f"Réponse de l'API : {data['message']}")
        print("L'index est en cours de reconstruction en arrière-plan...")
    else:
        print(f"Erreur {response.status_code} : {response.text}")

if __name__ == "__main__":
    # Assurez-vous que l'API est lancée (uvicorn src.api.main:app)
    
    test_ask_endpoint()
    
    # Décommentez pour tester la reconstruction
    # test_rebuild_endpoint()
    
    print("\nTests terminés.")