import google.generativeai as genai

API_KEY = os.getenv("API")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-3-flash-preview")

def generate_explanation(data):

    prompt = f"""
Explain this AI chest X-ray analysis.

Prediction: {data['prediction']}
Confidence: {data['confidence']}
Heart overlap: {data['heart_overlap_average']}
Agreement score: {data['agreement_score']}
Trust level: {data['trust_level']}
"""

    response = model.generate_content(prompt)

    return response.text
