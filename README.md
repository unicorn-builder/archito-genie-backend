# Archito-Genie FastAPI Backend

Backend API pour le générateur de design conceptuel Archito-Genie.

## Prérequis

- Python 3.9+
- Clé API OpenAI

## Installation locale

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Définir la clé OpenAI
export OPENAI_API_KEY="sk-your-key-here"

# Lancer le serveur
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Déploiement avec Docker

```bash
# Build
docker build -t archito-genie-api .

# Run
docker run -p 8000:8000 -e OPENAI_API_KEY="sk-your-key" archito-genie-api
```

## Déploiement sur Railway/Render

1. Créer un nouveau projet
2. Connecter ce dossier
3. Ajouter la variable d'environnement `OPENAI_API_KEY`
4. Déployer

## Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Health check |
| POST | `/projects` | Créer un nouveau projet |
| POST | `/projects/{id}/files` | Uploader les fichiers |
| POST | `/projects/{id}/analyze` | Lancer l'analyse technique |
| GET | `/projects/{id}/engineering-result` | Récupérer le résultat d'analyse |
| GET | `/projects/{id}/report` | Générer le rapport AI (OpenAI) |

## Configuration Frontend

Dans votre app Lovable, définissez l'URL du backend :

```env
VITE_FASTAPI_URL=https://your-deployed-api.com
```

Ou modifiez `src/config/api.ts`.

## Structure des réponses

### POST /projects

```json
{
  "id": "uuid",
  "name": "Mon Projet",
  "location": "Dakar, Sénégal",
  "building_type": "Residential - Multi-Family",
  "levels": 5,
  "gross_floor_area_m2": 3000,
  "climate_zone": "hot-humid",
  "target_certifications": ["EDGE", "LEED"],
  "files": [],
  "analyzed": false
}
```

### GET /projects/{id}/report

```json
{
  "project_id": "uuid",
  "report_markdown": "# Archito-Genie Design Report\n\n**Disclaimer:**..."
}
```
