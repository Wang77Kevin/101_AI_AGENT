import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
# client = OpenAI(
#     api_key="sk-KaWMoWOxGWpNFRDGitpCqPDChlO2VJT02fIbMfWsoKwpMG1w",
#     base_url="https://sg.uiuiapi.com/v1",
# )

response = client.responses.create(
    model="gpt-5-nano", input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
