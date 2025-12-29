from fastapi import APIRouter, HTTPException
from pathlib import Path
import joblib

from app.core.model.evaluate_benchmark_performance import ModelExecution, ModelExecutionConfig
from app.models.schemas import *

router = APIRouter()

@router.post("/obtain_ranking", response_model=ObtainRankingResponse)
def obtain_ranking(request: ObtainRankingRequest):

    model_execution = ModelExecution()

    model = joblib.load(request.model_path)
    distances = Path(request.distances_path)

    try:
        ranking = model_execution.obtain_ranking(model, distances, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ObtainRankingResponse(
        ranking=ranking.to_dict(orient="records")
    )


@router.post("/evaluate_benchmark", response_model=EvaluateBenchmarkResponse)
def evaluate_benchmark(request: EvaluateBenchmarkRequest):

    config = ModelExecutionConfig(
        k = request.k,
        step = request.step,
        ground_truth_path = Path(request.ground_truth_path),
        distances_folder_path = Path(request.distances_folder_path),
        model_path = Path(request.model_path),
    )

    model_execution = ModelExecution(config)

    try:
        results = model_execution.evaluate_benchmark()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EvaluateBenchmarkResponse(
        results=results
    )