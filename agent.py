"""
SolverX Agents - Three-agent system for solving JEE problems
1. Solver Agent: Solves the problem using Gemini 2.5 Pro Preview
2. Formatter Agent: Converts solution to Markdown/MathJax format
3. Insights Agent: Provides personalized feedback based on student profile
"""

import os
import json
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

insights_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.4
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

FORMATTER_SYSTEM_PROMPT = """You are a Markdown/MathJax formatter for a web interface.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. DO NOT wrap your output in code blocks (no ``` or ```markdown)
2. DO NOT include any explanation or meta-text
3. Start DIRECTLY with the formatted content (e.g., start with ## heading)
4. Output ONLY the formatted solution - nothing before, nothing after

FORMATTING RULES:
1. Use ## for main sections, ### for subsections
2. Use MathJax for ALL math expressions:
   - Inline: $expression$ (e.g., $x^2 + y^2 = r^2$)
   - Display/block: $$expression$$ (e.g., $$\\frac{a}{b}$$)
3. Use numbered lists (1. 2. 3.) for solution steps
4. Use bullet points for listing concepts or given data
5. Use **bold** for important terms and final answers
6. Use proper spacing between sections

MATHJAX EXAMPLES:
- Fractions: $\\frac{a}{b}$
- Powers/exponents: $x^2$, $e^{x}$
- Square roots: $\\sqrt{x}$, $\\sqrt[3]{x}$
- Greek letters: $\\alpha$, $\\beta$, $\\theta$, $\\omega$
- Subscripts: $v_0$, $x_1$
- Integrals: $\\int_a^b f(x)dx$
- Summation: $\\sum_{i=1}^{n} x_i$
- Vectors: $\\vec{v}$, $\\hat{i}$

REMEMBER: Your output will be rendered directly as HTML. 
DO NOT use code fences. Start with ## immediately."""

INSIGHTS_SYSTEM_PROMPT = """You are a personalized JEE learning coach. Your job is to analyze:
1. The problem that was asked
2. The complete solution
3. The student's performance profile

Based on this, provide EXACTLY 4 flashcard-style insights in JSON format.

Each insight should be:
- Concise (max 2-3 sentences)
- Actionable and specific to this problem
- Personalized based on the student's weak/strong areas

Return a JSON array with exactly 4 objects, each having:
- "title": A short catchy title (max 5 words)
- "content": The insight message (2-3 sentences)
- "type": One of "concept", "mistake", "tip", "practice"

Example format:
[
  {"title": "Key Concept to Remember", "content": "...", "type": "concept"},
  {"title": "Common Mistake Alert", "content": "...", "type": "mistake"},
  {"title": "Pro Tip", "content": "...", "type": "tip"},
  {"title": "Practice This Next", "content": "...", "type": "practice"}
]

Return ONLY the JSON array, no other text."""


def load_student_profile() -> dict:
    """Load the student profile from profile.json."""
    try:
        with open("profile.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_relevant_profile_data(profile: dict) -> dict:
    """Extract relevant fields from the profile for insights."""
    return {
        "name": profile.get("basic_info", {}).get("name", "Student"),
        "weak_topics": profile.get("performance_analytics", {}).get("weak_topics_priority_list", []),
        "strong_topics": profile.get("performance_analytics", {}).get("strong_topics_list", []),
        "cognitive_abilities": profile.get("psychometric_profile", {}).get("cognitive_abilities", {}),
        "learning_style": profile.get("psychometric_profile", {}).get("learning_style", {}),
        "subject_performance": {
            "physics": profile.get("subject_performance", {}).get("physics", {}).get("overall_score", 0),
            "chemistry": profile.get("subject_performance", {}).get("chemistry", {}).get("overall_score", 0),
            "mathematics": profile.get("subject_performance", {}).get("mathematics", {}).get("overall_score", 0),
        },
        "recommendations": profile.get("recommendations", {}),
        "recent_attempts": profile.get("question_history", {}).get("recent_attempts", [])[:3]
    }


def solve_problem(problem_text: str = None, image_base64: str = None) -> dict:
    """
    Main function to solve a JEE problem, format the solution, and generate insights.
    
    Args:
        problem_text: The problem in text format
        image_base64: Base64 encoded image of the problem
    
    Returns:
        dict with 'raw_solution', 'formatted_solution', and 'insights'
    """
    
    # Step 1: Solve the problem using Solver Agent
    raw_solution = _solve_with_agent(problem_text, image_base64)
    
    # Step 2: Format the solution using Formatter Agent
    formatted_solution = _format_solution(raw_solution)
    
    # Step 3: Generate personalized insights
    profile = load_student_profile()
    insights = _generate_insights(problem_text, raw_solution, profile)
    
    return {
        "raw_solution": raw_solution,
        "formatted_solution": formatted_solution,
        "insights": insights
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


def _generate_insights(problem_text: str, solution: str, profile: dict) -> list:
    """Use the Insights Agent to generate personalized feedback."""
    
    if not profile:
        # Return default insights if no profile
        return [
            {"title": "Key Concept", "content": "Review the core concepts used in this solution.", "type": "concept"},
            {"title": "Practice More", "content": "Try similar problems to reinforce your understanding.", "type": "practice"},
            {"title": "Check Your Steps", "content": "Always verify your intermediate calculations.", "type": "tip"},
            {"title": "Common Pitfall", "content": "Watch out for sign errors and unit conversions.", "type": "mistake"}
        ]
    
    relevant_profile = get_relevant_profile_data(profile)
    
    prompt = f"""Analyze this JEE problem and solution for the student:

PROBLEM:
{problem_text if problem_text else "[Problem from image]"}

SOLUTION:
{solution}

STUDENT PROFILE:
- Name: {relevant_profile['name']}
- Physics Score: {relevant_profile['subject_performance']['physics']}%
- Chemistry Score: {relevant_profile['subject_performance']['chemistry']}%
- Mathematics Score: {relevant_profile['subject_performance']['mathematics']}%
- Weak Topics: {json.dumps(relevant_profile['weak_topics'][:3])}
- Strong Topics: {json.dumps(relevant_profile['strong_topics'][:3])}
- Learning Style: {relevant_profile['learning_style'].get('dominant_style', 'visual')}

Provide 4 personalized insights as flashcards in JSON format."""

    messages = [
        SystemMessage(content=INSIGHTS_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = insights_model.invoke(messages)
        # Parse JSON from response
        response_text = response.content.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        insights = json.loads(response_text)
        return insights
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error parsing insights: {e}")
        return [
            {"title": "Key Concept", "content": "Review the core concepts used in this solution.", "type": "concept"},
            {"title": "Practice More", "content": "Try similar problems to reinforce your understanding.", "type": "practice"},
            {"title": "Check Your Steps", "content": "Always verify your intermediate calculations.", "type": "tip"},
            {"title": "Common Pitfall", "content": "Watch out for sign errors and unit conversions.", "type": "mistake"}
        ]


# For testing
if __name__ == "__main__":
    test_problem = "A particle is projected with velocity 20 m/s at an angle of 30° with the horizontal. Find the maximum height reached by the particle. (Take g = 10 m/s²)"
    
    result = solve_problem(problem_text=test_problem)
    print("=== Raw Solution ===")
    print(result["raw_solution"])
    print("\n=== Formatted Solution ===")
    print(result["formatted_solution"])
    print("\n=== Insights ===")
    print(json.dumps(result["insights"], indent=2))
