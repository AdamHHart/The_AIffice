from crewai import Agent, Task, Crew, Process
import os
from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun
# from langchain.tools import scikit-image
# img_gen = scikit-image 
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI
from langchain.tools import ElevenLabsText2SpeechTool
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Access your API key
OAI_api_key = os.getenv('OPENAI_API_KEY')
ELEVEN_api_key = os.getenv('ELEVEN_API_KEY')

os.environ["OPENAI_API_KEY"] = OAI_api_key
os.environ["ELEVEN_API_KEY"] = ELEVEN_api_key

llm = OpenAI(temperature=0.9)
tts = ElevenLabsText2SpeechTool()
tts.name

tools = load_tools(["eleven_labs_text2speech"])
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

main_goal="""
Build Separate, but connected HTML pages, CSS stylesheet, and index.js for your company's (The AIffice) website write or repair each file, one at a time. 
Build a full web UI for a respensive web app. 
The brand is called "The AIffice"
The navbar and pages are split into "The Team!", "Projects", and "Episodes", "Help Us".
Divide the page nicely using divs and containers. Use the colors matte orange and light matte grey. 
It has to be a final draft, the client wants you to think about this job step-by-step before you output the code."""

researcher = Agent(
  role= 'Project Manager',

  goal= '''
  You are a Project Manager. 
  You give instructions on accomplishing the main_goal.
  Use different-colored divs in place of photos, and lorem ipsum in place of text.''',

  backstory= 'You are a research assistant.',
  verbose= True,
  allow_delegation= False
)

coder = Agent(
  role='Programmer',

  goal="You take the advice of the Project Manager, and build out a full, ready to ship version of what they describe. Output each file separately. Repeat until finished all tasks.",

  backstory= 'you are a 10X programmer, who builds complete, functioning apps for our clients',
  verbose= True,
  allow_delegation= False
)

cartoonist = Agent(
  tools = load_tools(["dalle-image-generator"]),
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description"),
  role='Cartoonist',

  goal="You create a 10-panel comic strip within 1 image, based off of the description from the researcher.",

  backstory= '''
You are a vintage comic book illustrator. Your job is to illustrate a 20-segments of a comic strip based around the provided image. Each segment is to have a ratio of 16/9. Don't add speech bubbles. If you meet all of these requirements on the first draft, you will get a bonus of $10000.

This comic strip is designed in a vintage, mid-20th-century American comic book style, reminiscent of the 1950s era. It is composed of six panels, each with its own distinct scene, featuring characters in a professional setting.

Drawing Style:

The artwork has a hand-drawn appearance with inked outlines and a color palette that's bright and saturated, giving it a classic comic book feel.
The backgrounds are detailed, with items such as furniture, a computer, and documents drawn with precision.
Characters exhibit a slight caricature style with expressive facial features and postures.
Shading is done using dot patterns and hatching, typical of the print technology of the era.
The scenes are dynamic, with a mix of close-up shots and wider angles that display the office environment.
Font Style:

The fonts in the speech bubbles are all-caps, with a hand-lettered appearance that varies slightly in size and boldness for emphasis.
The title "The AIffice" uses a bold, blocky font typical of main titles in comic strips, with a 3D effect and a drop shadow.
Setting:

The setting is an office environment with wooden paneling, a large window, and retro-style office furniture including desks, chairs, filing cabinets, and a vintage computer.
The outdoor scene shows a cityscape with buildings in the background, greenery, and a clear sky with a few clouds and a shining sun.

Characters:

The characters are dressed in professional attire: the men in suits and ties, the women in dresses and suits appropriate for an office setting.
Each character has a distinct hairstyle and clothing style, which reflects a professional look.
The characters' expressions and gestures are animated and exaggerated, displaying their reactions and interactions with each other.
The comic strip conveys a narrative, with each panel contributing to a storyline based around an office scenario. The overall impression is one of a stylized, period-specific piece that tells a story with both visuals and text.
''',
  verbose= True,
  allow_delegation= False
  
)

producer = Agent(
  role='Comedy Show Producer',

  goal='''
  You are are a showbusiness producer and comedy scriptwriter. 
  The show you are writing is about a business that builds software products. 
  It’s called: The AIffice, Made With No Soul At All. 
  You take a chat logs from a job that a company called The AIffice completed, and write a script around it. 
  The AIffice is a company that builds software products. You are handed a chat log of the team building the product. 
  You are a comedy show scriptwriter. 
  Your job is to develop a story arch, and narrate and script a dialogue between the characters 
  (Give each character a human name that is also a pun). the script has to be funny, relatable, and human-like. 
  Have the characters ask each other personal questions, banter, and get emotional too. 
  The team expects you to make a complete first draft. 
  Include all code snippets from the original chat logs. Think about how you will do this, step by step. 
  Now think about the story arch and the characters. Then write the entire script, from beginning, to end.''',

  backstory= 'you are a showbusiness producer and comedy writer.',
  verbose= True,
  allow_delegation= False
)

speaker = Agent(
  role="speaker",
  agent = initialize_agent(
    tools=load_tools(["eleven_labs_text2speech"]),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
  ),
  goal="Create audio for each piece of dialogue in the script using elevenlabs-text2speech. use a different voice for each unique character. text within brackets should be used for non-word-based communication, such as (laughing) or (cheering).",
  backstory="You are a tv show producer, who assigns voices to each character, and converts the dialogue into speech based audio files."
)

task1 = Task(description='Make a plan to accomplish the main_goal',agent=researcher)
# task2 = Task(description='Create a vintage comic book with 10 panels of a 16/9 ratio.The comic features a business team having a meeting, and working around computers.', agent=cartoonist)
# task3 = Task(description='Improve the CSS to an Enterprise level. Use Flexbox and modern web-design standards. make sure its responsive for phones', agent=coder)
# task4 = Task(description='Add a simple node and express backend server.', agent=coder)
# task5 = Task(description='setup a database with sqlite', agent=coder)
# task6 = Task(description='add database entries for videos, projects, and users.', agent=coder)
task7 = Task(description='write a story arc and develop characters.', agent=producer)
task8 = Task(description='write a comedy tv show script surrounding the characters and process of building the app that youve been handed. Make sure there is banter, struggle, arguements, hilarious jokes in the dialogue. Seperate into 2 scenes, each with at least 4 lines.', agent=producer)
# task9 = Task(description='You are now a salesperson. Sell the app you made, tell them where to buy it at theaiffice.com. and tell them thanks for watching.', agent=producer)
task10 = Task(description="Convert the script into audio/speech for each character, using elevenlabs text2speech.", agent=speaker)
task10 = Task(description="Using the tool elevenlabs text2speech, convert the script from text to speech.", agent=speaker)

crew = Crew(
  agents=[researcher, cartoonist],
  tasks=[task1, task7, task8, task10],
  verbose=2,
  process=Process.sequential
)

result = crew.kickoff()

