from fastapi import FastAPI

from .routes import router

app = FastAPI(title="Paperwise AI", version="0.1.0")
app.include_router(router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}