# Local Development Setup

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose

## Step 1: Start Infrastructure Services

```bash
# Start PostgreSQL, Redis, MinIO, Milvus, etcd, Attu
docker-compose up -d

# Verify services are running
docker-compose ps
```

Services will be available at:
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **MinIO**: localhost:9000 (console: http://localhost:9001)
- **Milvus**: localhost:19530
- **Attu** (Milvus UI): http://localhost:8000

## Step 2: Setup Backend

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env

# Edit .env with your API keys (optional for local dev)
# You can also set API keys per-agent via the UI later

# Run database migrations
python -m alembic upgrade head

# Create test users (see script below)
python scripts/create_test_users.py

# Start backend
python -m uvicorn ai_agent.main:app --reload --port 8000
```

Backend will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs

## Step 3: Setup Frontend

```bash
cd ../pb-ai-agent-41

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend will be available at: http://localhost:5173

## Step 4: Test User Login

Use these credentials to test:

| Email | Password | Role |
|-------|----------|------|
| admin@example.com | admin123 | Admin |
| user@example.com | user123 | Regular User |
| test@example.com | test123 | Test User |

## Common Issues

### Database Connection Error
If you see `could not translate host name "db"`, make sure:
1. You're using `.env` with `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agentiq`
2. PostgreSQL container is running: `docker-compose ps`

### MinIO Bucket Not Found
The bucket is auto-created on startup, but if missing:
1. Go to http://localhost:9001
2. Login with minioadmin/minioadmin
3. Create bucket named `agentiq`

### API Key Issues
You can either:
1. Set global API keys in `.env` file
2. Set per-agent API keys via the Settings UI (recommended)

## Database Migrations

```bash
# Create new migration
python -m alembic revision --autogenerate -m "description"

# Apply migrations
python -m alembic upgrade head

# Downgrade
python -m alembic downgrade -1
```

## Reset Everything

```bash
# Stop all containers
docker-compose down -v

# Remove all data
docker-compose down -v --rmi all

# Start fresh
docker-compose up -d
python -m alembic upgrade head
python scripts/create_test_users.py
```
