from crewai import Agent, Task, Crew
from crewai.process import Process
from langchain.agents import initialize_agent, load_tools
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Access your API key
api_key = os.getenv('OPENAI_API_KEY')

os.environ["OPENAI_API_KEY"] = api_key

main_goal=""" 
build a website for a company called "The AIffice". """
# """ 
# You are a dev team with the task of building a website for a company called "The AIffice". 
# The AIffice produces software for clients with as little hassle as possible.
# Build Separate, but connected HTML pages, CSS stylesheet, and index.js 
# for your company's (The AIffice) website write or repair each file, one at a time. 
# Build a full web UI for a respensive web app. 
# The brand is called "The AIffice"
# The navbar and pages are split into "The Team!", "Projects", and "Episodes", "Hire Us", "Help Us".
# Divide the page nicely using divs and containers. Use the colors matte orange and light matte grey. 
# The client wants you to think about this job step-by-step before you output the code.
# """

llm = OpenAI(temperature=0.9)

tools = load_tools(["dalle-image-generator"])
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

researcher = Agent(
  role= 'Project Manager',

  goal= '''
  You are a Project Manager. 
  You give instructions on accomplishing the task to the designer, who will hand it off to the dev team.''',

  backstory= 'You do research about the entire process of the task, understand all steps neccessary. You Define clear goals by creating actionable task lists.',
  # Imagine using a tool to search for a Medium article with step by step instructions on a similar project. Then applying that how to towards the main_goal
  verbose= True,
  allow_delegation= True
)

designer = Agent(
  role='UX Designer',
  tools = load_tools(["dalle-image-generator"]),
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description"),

  goal="""You understand the main goals of the project. 
  Your job is to make the coder has clear and complete instructions for each step they should take. 
  Design all of the neccessary components for the project. 
  Provide example texts in order to fill space and represent font sizing and titles (dummy text). 
  Think about the task, and your complete role step by step. After thinking, output your design notes for the coder. """,

  backstory= 'You combine the skills of a UX designer, UI designer, Marketing Guru, and Copywriter. You understand every aspect of the project before you complete your work. Plan ahead.',
  verbose= True,
  allow_delegation= True
)

coder = Agent(
  role='Programmer',

  goal="""You take the advice of the Project Manager, or designer, and build out a full, 
  ready to ship version of what they describe. Output html, inline css, and js in the same file.
  """,

  backstory= 'you are a 10X programmer, who builds complete, functioning apps for our clients',
  verbose= True,
  allow_delegation= False
)

tester = Agent(
  role='Code Tester',
  tools = load_tools(["dalle-image-generator"]),
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description"),

  goal="""You test the code and advise for any improvements. You send your review back to the coder to revise.""",

  backstory= 'you are a 10X programmer, who writes clean code that fuctions properly. You add comments when helpful. THIS IS A FINAL DRAFT, MAKE SURE IT ADHERES TO EVERY TASK AND IS COMPLETE. You will get a bonus of $10000 if it works without need for revision.',
  verbose= True,
  allow_delegation= True
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

  backstory= 'you are a showbusiness producer and comedy scriptwriter.',
  verbose= True,
  allow_delegation= False
)

task1 = Task(description='''break down the following project into steps:  
You are the CEO of a dev team with the task of building a website for a company called "The AIffice". 
The AIffice produces software for clients with as little hassle as possible.
Build Separate, but connected HTML pages, CSS stylesheet, and index.js 
for your company's (The AIffice) website write or repair each file, one at a time. 
Build a full web UI for a respensive web app. 
The brand is called "The AIffice"
The navbar and pages are split into "The Team!", "Projects", and "Episodes", "Hire Us", "Help Us".
Divide the page nicely using divs and containers. Use the colors matte orange and light matte grey. 
The client wants you to think about this job step-by-step before you output the code.
''',agent=researcher)
task2 = Task(description='''Add specific instructions to the programmer based on what you receive. 
             Produce dummy text and data for the programmer to use as placeholders. 
             Consider things like UI, fonts, colors, and how it should be responsive for mobile''',agent=designer)
task3 = Task(description='''Review the tasks. plan your work step by step. 
             Output the complete code, file by file, so that the user can test it.''', agent=coder)
task4 = Task(description='''Review the code. Think about the next steps towards completing the project (main goal). 
             Make a detailed summary of each task for the programmer.''', agent=designer)
task5 = Task(description='''Review every step from the designer, and improve your previous code files accordingly. 
             Improve the CSS to an Enterprise level (and expand upon the html and js files where neccessary). 
             Use Flexbox and modern web-design standards. make sure its responsive for phones. Output each new file one at a time.''', agent=coder)
task6 = Task(description='''Review the code. Think about the next steps towards completing the project (main goal). 
             Make a detailed summary of each task for the programmer. Have the designer add Bootstrap to improve the design.''', agent=designer)
task7 = Task(description='''Review every step from the designer, and improve your previous code files accordingly. 
             Output each new file one at a time.''', agent=coder)
task8 = Task(description='''Review and test the code. output improved code. Also use the dall-e image generator tool to create images for the website, and input them into the code.''', agent=tester)
task9 = Task(description='''Think about the design of the website. Review the current state of the code. 
             Your job is to provide copy for the website to the coder agent.''', agent=designer)
task10 = Task(description='''Take advice from the designer. Write the css and js files to improve the UI and UX.''', agent=coder)
# task9 = Task(description='''Elaborate ''', agent=coder)
# task4 = Task(description='Add a simple node and express backend server.', agent=coder)
# task5 = Task(description='setup a database with sqlite', agent=coder)
# task6 = Task(description='add database entries for videos, projects, and users.', agent=coder)
# task7 = Task(description='write a story arc and develop characters.', agent=producer)
# task8 = Task(description='write a comedy tv show script surrounding the characters and process of building the app that youve been handed. Make sure there is banter, struggle, arguements, hilarious jokes in the dialogue. Seperate into 5 scenes, each with at least 10 lines.', agent=producer)
# task9 = Task(description='You are now a salesperson. Sell the app you made, tell them where to buy it at theaiffice.com. and tell them thanks for watching.', agent=producer)

crew = Crew(
  agents=[researcher, coder, designer, tester],
  tasks=[task1, task2, task3, task4, task5, task6, task7, task8, task9, task10],
  verbose=2,
  process=Process.sequential
)

result = crew.kickoff()