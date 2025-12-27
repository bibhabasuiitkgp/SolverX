"""
SolverX Backend - FastAPI server for the JEE problem solver
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64

from agent import solve_problem

app = FastAPI(title="SolverX API", description="JEE Problem Solver API")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextProblemRequest(BaseModel):
    problem_text: str


class SolutionResponse(BaseModel):
    raw_solution: str
    formatted_solution: str


@app.get("/")
async def root():
    return {"message": "SolverX API is running!", "status": "healthy"}


@app.post("/solve/text", response_model=SolutionResponse)
async def solve_text_problem(request: TextProblemRequest):
    """Solve a JEE problem provided as text."""
    result = solve_problem(problem_text=request.problem_text)
    return SolutionResponse(**result)


@app.post("/solve/image", response_model=SolutionResponse)
async def solve_image_problem(
    image: UploadFile = File(...),
    additional_text: Optional[str] = Form(None)
):
    """Solve a JEE problem from an uploaded image."""
    # Read and encode the image
    image_bytes = await image.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    result = solve_problem(
        problem_text=additional_text,
        image_base64=image_base64
    )
    return SolutionResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
