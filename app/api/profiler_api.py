from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.core.profiling.profiler import DataProfiler, ProfilerConfig
from app.models.schemas import ProfileRequest, ProfileResponse

router = APIRouter()

@router.post("/profile_datalake", response_model=ProfileResponse)
def generate_profiles_for_datalake(request: ProfileRequest):

    config = ProfilerConfig(
        datalake_path=Path(request.datalake_path),
        output_profiles_path=Path(request.output_profiles_path),
        max_workers=request.max_workers,
        varchar_only=request.varchar_only,
    )

    profiler = DataProfiler(config)

    try:
        status = profiler.generate_profiles_for_datalake()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ProfileResponse(
        status=status
    )