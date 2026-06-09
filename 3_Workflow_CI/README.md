# Kriteria 3 - Workflow CI dengan MLProject dan Docker Hub

Folder ini berisi MLProject untuk retraining otomatis melalui GitHub Actions dan build Docker image ke Docker Hub.

Test lokal:

```bash
cd MLProject
mlflow run . --env-manager=local
```

Atau dari folder `3_Workflow_CI`:

```bash
mlflow run .\MLProject --env-manager=local
```
