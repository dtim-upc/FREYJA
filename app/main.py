import app.logging_config # Do not remove

## uvicorn app.main:app --reload
## Docs -> http://localhost:8000/docs

from fastapi import FastAPI
from app.api.profiler_api import router as profiler_router
from app.api.distance_calculation_api import router as distances_router
from app.api.model_execution_api import router as model_router

app = FastAPI(title="FREYJA")

app.include_router(profiler_router, prefix="/profiles", tags=["Profiles"])
app.include_router(distances_router, prefix="/distances", tags=["Distances"])
app.include_router(model_router, prefix="/model", tags=["Model"])