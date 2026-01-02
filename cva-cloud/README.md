# CVA Cloud

SaaS console for Catalyst Vector Alpha.

## Structure
- `apps/web`: Next.js 14 dashboard
- `apps/api`: FastAPI backend
- `packages/shared`: shared types

## Quick start
1. Start services:
   ```bash
   docker compose up -d
   ```
2. API setup:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r apps/api/requirements.txt
   ```
3. Run migrations (after you add Alembic revisions):
   ```bash
   cd apps/api
   alembic revision --autogenerate -m "init"
   alembic upgrade head
   ```
4. Seed demo data:
   ```bash
   python apps/api/scripts/seed_demo.py
   ```
5. Run API:
   ```bash
   cd apps/api
   uvicorn app.main:app --reload --port 8000
   ```
6. Run web:
   ```bash
   cd apps/web
   npm install
   npm run dev
   ```

## Notes
- Auth is scaffolded for Clerk; add your Clerk env vars in `apps/web/.env.local`.
- API reads org context from `X-Org-Id` header until JWT validation is wired.
