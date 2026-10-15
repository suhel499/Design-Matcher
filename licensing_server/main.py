"""
FastAPI Server Entry Point for Design Matcher Licensing & Subscription Authority.
Serves public licensing API, admin management endpoints, and responsive Admin Portal.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure design_matcher is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "design_matcher"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from licensing_server.config import settings
from licensing_server.database import Base, SessionLocal, engine
from licensing_server.models import AdminUser, Product
from licensing_server.routes.admin import router as admin_router
from licensing_server.routes.licenses import router as licenses_router
from licensing_server.security import hash_password

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_STATIC_DIR = Path(__file__).resolve().parent / "admin_portal" / "static"
_TEMPLATE_DIR = Path(__file__).resolve().parent / "admin_portal" / "templates"


def init_db_and_seeds() -> None:
    """Creates database tables and seeds initial product and superadmin if missing."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # 1. Seed Product
        prod = db.query(Product).filter(Product.product_code == settings.DEFAULT_PRODUCT_CODE).first()
        if not prod:
            prod = Product(
                product_code=settings.DEFAULT_PRODUCT_CODE,
                name=settings.DEFAULT_PRODUCT_NAME,
                description="Desktop CAD/DXF and PDF Design Matching Suite",
            )
            db.add(prod)
            logger.info("Seeded default product: %s", settings.DEFAULT_PRODUCT_CODE)

        # 2. Seed Admin User
        admin = db.query(AdminUser).filter(AdminUser.username == settings.INITIAL_ADMIN_USERNAME).first()
        if not admin:
            admin = AdminUser(
                username=settings.INITIAL_ADMIN_USERNAME,
                email=settings.INITIAL_ADMIN_EMAIL,
                password_hash=hash_password(settings.INITIAL_ADMIN_PASSWORD),
                role="superadmin",
                is_active=True,
            )
            db.add(admin)
            logger.info("Seeded initial administrator account: %s", settings.INITIAL_ADMIN_USERNAME)

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Error during startup seed initialization: %s", exc)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Starting up %s...", settings.APP_NAME)
    init_db_and_seeds()
    yield
    logger.info("Shutting down %s...", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Cryptographically authoritative licensing and subscription manager for Design Matcher.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files for Admin Portal
if _STATIC_DIR.exists():
    app.mount("/admin-static", StaticFiles(directory=str(_STATIC_DIR)), name="admin-static")

# Include API Routers
app.include_router(licenses_router, prefix=settings.API_V1_PREFIX)
app.include_router(admin_router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
def health_check():
    """Service health and liveness probe."""
    return {"status": "ok", "app": settings.APP_NAME, "version": "1.0.0"}


@app.get("/admin", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def serve_admin_dashboard():
    """Serves the interactive single-page Admin Web Portal."""
    index_file = _TEMPLATE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>Admin portal template not found.</h2>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("licensing_server.main:app", host="0.0.0.0", port=8000, reload=True)
