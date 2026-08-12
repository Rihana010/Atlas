import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app

def test_api_endpoints():
    client = TestClient(app)
    
    # 1. Test Root
    res = client.get("/")
    assert res.status_code == 200
    print("GET / ->", res.json())

    # 2. Test Upload API
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "sample_classification.csv"))
    with open(csv_path, "rb") as f:
        res_upload = client.post("/upload", files={"file": ("sample_classification.csv", f, "text/csv")})
    
    assert res_upload.status_code == 200
    upload_data = res_upload.json()
    exp_id = upload_data["experiment_id"]
    print("POST /upload ->", upload_data)

    # 3. Test Dataset Info API
    res_dataset = client.get(f"/dataset/{exp_id}")
    assert res_dataset.status_code == 200
    dataset_data = res_dataset.json()
    print(f"GET /dataset/{exp_id} -> Total Columns:", dataset_data["total_columns"])

    # 4. Test Train API
    res_train = client.post("/train", json={"experiment_id": exp_id, "target_column": "Survived"})
    assert res_train.status_code == 200
    train_data = res_train.json()
    print("POST /train ->", train_data)

    # 5. Poll Results API until completed
    print("Polling GET /results/{exp_id}...")
    completed = False
    for _ in range(20):
        time.sleep(0.5)
        res_results = client.get(f"/results/{exp_id}")
        assert res_results.status_code == 200
        results_data = res_results.json()
        print(f"Status: {results_data['status']} | Best Model: {results_data.get('best_model')}")
        if results_data["status"] in ["completed", "failed"]:
            completed = True
            break
            
    assert completed, "Training pipeline should complete within timeout."
    print("API ENDPOINTS TEST PASSED!")

if __name__ == "__main__":
    test_api_endpoints()
