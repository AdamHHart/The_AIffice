import os

from langchain_openai import OpenAI
from langchain.agents import initialize_agent, load_tools
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Access your API key
api_key = os.getenv('OPENAI_API_KEY')

os.environ["OPENAI_API_KEY"] = api_key

llm = OpenAI(temperature=0.9)

tools = load_tools(["dalle-image-generator"])
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
output = agent.run("Create an image of a halloween night at a haunted museum")