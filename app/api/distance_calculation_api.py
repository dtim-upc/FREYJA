from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.core.distances.compute_distances import ComputeDistances, ComputeDistancesConfig
from app.models.schemas import DistancesRequest, DistancesResponse, DistancesForQueryRequest

router = APIRouter()

@router.post("/distances_for_query", response_model=DistancesResponse)
def compute_distances_for_query(request: DistancesForQueryRequest):

    config = ComputeDistancesConfig(
        profiles_file_path=Path(request.profiles_file_path),
        ground_truth_path=None,
        output_distances_path=Path(request.output_distances_path)
    )

    distance_computer = ComputeDistances(config)
    distance_computer.query_attribute = request.query_column
    distance_computer.query_dataset = request.query_dataset

    try:
        status = distance_computer.generate_distances_for_query()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DistancesResponse(
        status=status
    )

@router.post("/distances_for_benchmark", response_model=DistancesResponse)
def generate_distances_for_benchmark(request: DistancesRequest):

    config = ComputeDistancesConfig(
        profiles_file_path=Path(request.profiles_file_path),
        ground_truth_path=Path(request.ground_truth_path),
        output_distances_path=Path(request.output_distances_path)
    )

    distance_computer = ComputeDistances(config)

    try:
        status = distance_computer.generate_distances_for_benchmark()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DistancesResponse(
        status=status
    )

@router.post("/distances_for_training_model", response_model=DistancesResponse)
def generate_distances_for_training_model(request: DistancesRequest):

    config = ComputeDistancesConfig(
        profiles_file_path=Path(request.profiles_file_path),
        ground_truth_path=Path(request.ground_truth_path),
        output_distances_path=Path(request.output_distances_path)
    )

    distance_computer = ComputeDistances(config)

    try:
        status = distance_computer.generate_distances_for_training_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DistancesResponse(
        status=status
    )