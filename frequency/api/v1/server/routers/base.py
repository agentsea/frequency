# generated by fastapi-codegen:
#   filename:  server.yaml
#   timestamp: 2024-01-13T18:45:42+00:00

from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import *

router = APIRouter(tags=['Base'])


@router.get('/v1', response_model=V1Info, tags=['Base'])
def get_root() -> V1Info:
    """
    API info
    """
    pass


@router.get('/v1/health', response_model=V1Health, tags=['Base'])
def get_health() -> V1Health:
    """
    Health info
    """
    pass
