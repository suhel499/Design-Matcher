"""
API routes for Licensing Server.
"""

from licensing_server.routes.admin import router as admin_router
from licensing_server.routes.licenses import router as licenses_router

__all__ = ["admin_router", "licenses_router"]
