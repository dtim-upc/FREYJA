from pydantic import BaseModel

class ProfileRequest(BaseModel):
    datalake_path: str
    output_profiles_path: str
    varchar_only: bool = True
    max_workers: int = 8

class ProfileResponse(BaseModel):
    status: str

############################################

class DistancesRequest(BaseModel):
    profiles_file_path: str
    ground_truth_path: str
    output_distances_path: str

class DistancesResponse(BaseModel):
    status: str

class DistancesForQueryRequest(BaseModel):
    profiles_file_path: str
    output_distances_path: str
    query_column: str
    query_dataset: str

############################################

class ObtainRankingRequest(BaseModel):
    model_path: str
    distances_path: str
    top_k: int

class ObtainRankingResponse(BaseModel):
    ranking: list[dict]

class EvaluateBenchmarkRequest(BaseModel):
    k: int
    step: int
    ground_truth_path: str
    distances_folder_path: str
    model_path: str

class EvaluateBenchmarkResponse(BaseModel):
    results: dict

    