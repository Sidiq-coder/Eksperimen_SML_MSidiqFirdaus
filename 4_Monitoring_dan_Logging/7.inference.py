import json
import requests
from requests.exceptions import ConnectionError, Timeout

FEATURES_URL = "http://localhost:8000/features"
PREDICT_URL = "http://localhost:8000/predict"


def main():
    try:
        feature_response = requests.get(FEATURES_URL, timeout=10)
    except (ConnectionError, Timeout) as exc:
        raise SystemExit(
            "API monitoring belum berjalan di http://localhost:8000. "
            "Jalankan dulu: python 3.prometheus_exporter.py"
        ) from exc

    feature_response.raise_for_status()
    features = feature_response.json()["features"]

    # Contoh payload preprocessed.
    # Nilai 0 dipakai sebagai baseline aman untuk memicu request monitoring.
    payload = {feature: 0 for feature in features}

    response = requests.post(PREDICT_URL, json=payload, timeout=10)
    print("Status code:", response.status_code)
    print(json.dumps(response.json(), indent=4))


if __name__ == "__main__":
    main()
