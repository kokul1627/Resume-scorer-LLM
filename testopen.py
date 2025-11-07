from google import genai

client = genai.Client(
    api_key="AIzaSyDIQ3KytxmTBSAUryx-2l5bHzBA6q7ZsyA"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="88+22",
)

print(response.text)