import requests

def test_web_server_get():
    response = requests.get("http://localhost:5757/predict")

    assert response.status_code == 200
    assert response.json() == {"y_pred": 2}

def test_web_server_post():
    response = requests.post(
            "http://localhost:5757/predict",
            json={
                "size": 120,
                "nb_rooms": 5,
                "garden": 0
            }
        )

    assert response.status_code == 200
    response_data = response.json()
    assert "y_pred" in response_data
