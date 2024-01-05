# generated by fastapi-codegen:
#   filename:  server.yaml
#   timestamp: 2024-01-05T05:18:11+00:00

from __future__ import annotations

from fastapi import FastAPI

from .routers import adapter, base, model

app = FastAPI(
    version='1.0.0',
    title='Frequency API',
    description='API for Frequency',
)

app.include_router(adapter.router)
app.include_router(base.router)
app.include_router(model.router)


@app.get("/")
async def root():
    return {"message": "Gateway of the App"}