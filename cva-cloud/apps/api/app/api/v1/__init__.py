from fastapi import APIRouter

from app.api.v1.routes import clusters, incidents

api_router = APIRouter()
api_router.include_router(clusters.router)
api_router.include_router(incidents.router)
