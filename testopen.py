from google import genai

client = genai.Client(
    api_key="your api key"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="88+22",
)

print(response.text)
