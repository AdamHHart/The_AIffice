from crewai import Agent, Task, Crew, Process
import os
from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun
# from langchain.tools import scikit-image
# img_gen = scikit-image 
from langchain.agents import initialize_agent, load_tools
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Access your API key
api_key = os.getenv('OPENAI_API_KEY')

os.environ["OPENAI_API_KEY"] = api_key

tools = load_tools(["dalle-image-generator"])
# from langchain_community.tools import 
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
  allow_delegation= True
)

coder = Agent(
  role='Programmer',

  goal="You take the advice of the Project Manager, and build out a full, ready to ship version of what they describe. Output each file separately. Repeat until finished all tasks.",

  backstory= 'you are a 10X programmer, who builds complete, functioning apps for our clients',
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
  allow_delegation= True
)

task1 = Task(description='Make a plan to accomplish the main_goal',agent=researcher)
task2 = Task(description='Output the complete code, so that the user can test it.', agent=coder)
task3 = Task(description='Improve the CSS to an Enterprise level. Use Flexbox and modern web-design standards. make sure its responsive for phones', agent=coder)
task4 = Task(description='Add a simple node and express backend server.', agent=coder)
task5 = Task(description='setup a database with sqlite', agent=coder)
task6 = Task(description='add database entries for videos, projects, and users.', agent=coder)
task7 = Task(description='write a story arc and develop characters.', agent=producer)
task8 = Task(description='write a comedy tv show script surrounding the characters and process of building the app that youve been handed. Make sure there is banter, struggle, arguements, hilarious jokes in the dialogue. Seperate into 5 scenes, each with at least 10 lines.', agent=producer)
task9 = Task(description='You are now a salesperson. Sell the app you made, tell them where to buy it at theaiffice.com. and tell them thanks for watching.', agent=producer)

crew = Crew(
  agents=[researcher, coder, producer],
  tasks=[task1, task2, task3, task4, task5, task6, task7, task8, task9],
  verbose=2,
  process=Process.sequential
)

result = crew.kickoff()