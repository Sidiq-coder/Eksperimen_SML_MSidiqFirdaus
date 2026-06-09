# Kriteria 4 - Monitoring dan Logging

Folder ini berisi konfigurasi serving, Prometheus, Grafana, exporter, dan inference request.

Jalankan exporter:

```bash
pip install -r requirements.txt
python 3.prometheus_exporter.py
```

Jalankan Prometheus dan Grafana:

```bash
docker compose up -d
```

Metric endpoint:

```text
http://localhost:8000/metrics
```
