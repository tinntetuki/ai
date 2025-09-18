"""
Main FastAPI application
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .video_upscale import router as upscale_router
from .speech_to_text import router as speech_router
from .text_to_speech import router as tts_router
from .product_detection import router as product_router


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="AI Content Creator",
        description="AI-powered video and image content creation platform",
        version="0.1.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(upscale_router)
    app.include_router(speech_router)
    app.include_router(tts_router)
    app.include_router(product_router)

    # Serve static files
    output_dir = Path(os.getenv("OUTPUT_PATH", "./data/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/downloads", StaticFiles(directory=str(output_dir)), name="downloads")

    # Serve frontend static files
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML page"""
        frontend_file = Path(__file__).parent.parent / "frontend" / "index.html"
        if frontend_file.exists():
            return FileResponse(str(frontend_file))
        else:
            return {"message": "Frontend not available, use /docs for API documentation"}

    @app.get("/api")
    async def api_root():
        return {
            "message": "AI Content Creator API",
            "version": "0.1.0",
            "endpoints": [
                "/api/v1/upscale/upload",
                "/api/v1/upscale/status/{task_id}",
                "/api/v1/upscale/download/{task_id}",
                "/api/v1/upscale/models",
                "/api/v1/speech/transcribe",
                "/api/v1/speech/status/{task_id}",
                "/api/v1/speech/download/{task_id}",
                "/api/v1/speech/models",
                "/api/v1/tts/synthesize",
                "/api/v1/tts/voiceover-from-video",
                "/api/v1/tts/status/{task_id}",
                "/api/v1/tts/download/{task_id}/{output_type}",
                "/api/v1/tts/voices",
                "/api/v1/tts/voice-profiles",
                "/api/v1/products/detect",
                "/api/v1/products/annotate",
                "/api/v1/products/status/{task_id}",
                "/api/v1/products/analysis/{task_id}",
                "/api/v1/products/download/{task_id}/{output_type}",
                "/api/v1/products/models",
                "/api/v1/products/categories"
            ]
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=debug,
        access_log=True
    )