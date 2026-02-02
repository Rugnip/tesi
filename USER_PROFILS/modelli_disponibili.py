from google import genai

client = genai.Client(api_key="AIzaSyCpgWjNDycIZTHKRI7z8mR54KA26pE8Sh0")

for model in client.models.list():
    print(model.name)
