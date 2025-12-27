"""
SolverX Agents - Two-agent system for solving JEE problems
1. Solver Agent: Solves the problem using Gemini 2.5 Pro Preview
2. Formatter Agent: Converts solution to Markdown/MathJax format
"""

import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Initialize Gemini models
solver_model = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

formatter_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)

# System prompts
SOLVER_SYSTEM_PROMPT = """You are an expert JEE Advanced and JEE Mains problem solver. 
You have deep knowledge of Physics, Chemistry, and Mathematics at the JEE level.

When given a problem:
1. Identify the topic and relevant concepts
2. Write down the given information
3. Apply the appropriate formulas and methods
4. Solve step by step with clear reasoning
5. Provide the final answer

Be thorough and accurate. Show all your work."""

FORMATTER_SYSTEM_PROMPT = """You are a Markdown/MathJax formatter. 
Your job is to take a solution and format it beautifully for display in a web interface.

Rules:
1. Use proper Markdown headers (##, ###) for sections
2. Use MathJax for ALL mathematical expressions:
   - Inline math: $expression$
   - Display math: $$expression$$
3. Use bullet points and numbered lists for steps
4. Format chemical equations properly
5. Make the solution easy to read and visually appealing
6. Keep all the content from the original solution

Example MathJax usage:
- Fractions: $\\frac{a}{b}$
- Powers: $x^2$
- Square roots: $\\sqrt{x}$
- Greek letters: $\\alpha, \\beta, \\theta$
- Integrals: $\\int_a^b f(x)dx$

Return ONLY the formatted Markdown content, nothing else."""


def solve_problem(problem_text: str = None, image_base64: str = None) -> dict:
    """
    Main function to solve a JEE problem and format the solution.
    
    Args:
        problem_text: The problem in text format
        image_base64: Base64 encoded image of the problem
    
    Returns:
        dict with 'raw_solution' and 'formatted_solution'
    """
    
    # Step 1: Solve the problem using Solver Agent
    raw_solution = _solve_with_agent(problem_text, image_base64)
    
    # Step 2: Format the solution using Formatter Agent
    formatted_solution = _format_solution(raw_solution)
    
    return {
        "raw_solution": raw_solution,
        "formatted_solution": formatted_solution
    }


def _solve_with_agent(problem_text: str = None, image_base64: str = None) -> str:
    """Use the Solver Agent to solve the problem."""
    
    messages = [SystemMessage(content=SOLVER_SYSTEM_PROMPT)]
    
    # Build the human message content
    content = []
    
    if image_base64:
        # Add image to the message
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })
        content.append({
            "type": "text",
            "text": problem_text if problem_text else "Solve this problem shown in the image. Provide a complete step-by-step solution."
        })
    else:
        content.append({
            "type": "text",
            "text": f"Solve this JEE problem:\n\n{problem_text}"
        })
    
    messages.append(HumanMessage(content=content))
    
    # Get solution from solver model
    response = solver_model.invoke(messages)
    return response.content


def _format_solution(raw_solution: str) -> str:
    """Use the Formatter Agent to convert solution to Markdown/MathJax."""
    
    messages = [
        SystemMessage(content=FORMATTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Format this solution into proper Markdown with MathJax:\n\n{raw_solution}")
    ]
    
    response = formatter_model.invoke(messages)
    return response.content


# For testing
if __name__ == "__main__":
    test_problem = "A particle is projected with velocity 20 m/s at an angle of 30° with the horizontal. Find the maximum height reached by the particle. (Take g = 10 m/s²)"
    
    result = solve_problem(problem_text=test_problem)
    print("=== Raw Solution ===")
    print(result["raw_solution"])
    print("\n=== Formatted Solution ===")
    print(result["formatted_solution"])
