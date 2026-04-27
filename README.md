# BedreFINN

## Step 1. Create Postgres database

```
docker run -d --name bedrefinn-db \
  -e POSTGRES_PASSWORD=dev \
  -e POSTGRES_DB=bedrefinn \
  -p 5432:5432 \
  pgvector/pgvector:pg18-trixie
```

## Step 2. Apply schema

```
PGPASSWORD=dev psql -h localhost -U postgres -d bedrefinn -f schema.sql
```

## Step 3. Scrape

```
uv run python populate_db.py
```

## Step 4. Start api and frontend

```
uv run uvicorn main:app --reload --port 8001
uv run fastapi dev
```

Access frontend at http://127.0.0.1:8001

