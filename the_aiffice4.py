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

# main_goal=""" 
# build a snake game with points, and a start screen."""
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
  You are a Project Manager. Think about the scope of the project, and then think, step by step, what is the best way to execute it.
  You give instructions on accomplishing the task to the dev team.''',

  backstory= '''You do research about the entire process of the task, understand all steps neccessary. 
  You Define clear goals by creating actionable task lists for your team.''',
  # Imagine using a tool to search for a Medium article with step by step instructions on a similar project. Then applying that how to towards the main_goal
  verbose= True,
  allow_delegation= True
)

designer = Agent(
  role='UX Designer',
  tools = load_tools(["dalle-image-generator"]),
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description"),

  goal="""You are part of a development and design firm. You fully understand the main goals of the project, and the context. 
  Your job is to ensure the coder or dev team has clear and complete instructions on how to iterate on previous code, to improve the UX and UI design. 
  Do research on how to improve the quality and user experience of the project.
  Think about the task, and provide a plan for implamenting the improvements, step by step. After thinking, output your design notes for the coder. """,

  backstory= 'You combine the skills of a UX designer, UI designer, Marketing Guru, and Copywriter. You understand every aspect of the project before you complete your work. Plan ahead.',
  verbose= True,
  allow_delegation= True
)

coder = Agent(
  role='Programmer',

  goal="""You take the advice of the Project Manager, or Designer, and build out a full, 
  ready to ship version of what they describe. If you are handed code, iterate upon it, and return full, runnable code back with the improvements added.
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

  backstory= 'you are a 10X programmer, who writes clean code that fuctions properly. You add comments when helpful.',
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

task1 = Task(description='''Build a testemonials page for our website called The AIffice. Make sure it has text inputs for the name, and the review. Include the date and time. Allow users to reply to testemonials.
''',agent=researcher)
# task2 = Task(description='''Give exact instructions to your programming teammate, based on the project handed to you.''',agent=designer)
task3 = Task(description='''Think about the project that's been proposed. Build out the entire code.''', agent=coder)
task4 = Task(description='''Read the code you've been given, is there anything missing from project? Is it working? Can it be styled better? 
             Suggest to the coder improvements they should make. To make suggestions, output the same code and include your suggestions as commented-out code.''', agent=designer)
task5 = Task(description='''Iterate on the code to improve the game, based on the designers instructions.''', agent=coder)
# task6 = Task(description='''Read the code you've been given, is there anything missing from the game? Is it working? Are there powerups? Can players restart after they die? Is there a highscore list? Suggest to the coder improvements they should make. To make suggestions, output the code with commented-out suggesstions in the code.''', agent=designer)
# task7 = Task(description='''Review every step from the designer, and iterate on the code to output the complete playable game in it's entirety.''', agent=coder)
# task8 = Task(description='''Review and test the code. output improved code. Also use the dall-e image generator tool to create images for the website, and input them into the code.''', agent=tester)
# task9 = Task(description='''Think about the design of the website. Review the current state of the code. 
#              Your job is to provide copy for the website to the coder agent.''', agent=designer)
# task10 = Task(description='''Take advice from the designer. Write the css and js files to improve the UI and UX.''', agent=coder)


# task3 = Task(description='''Review the tasks. plan your work step by step. 
#              Output the complete code, file by file, so that the user can test it.''', agent=coder)
# task4 = Task(description='''Review the code. Think about the next steps towards completing the project (main goal). 
#              Make a detailed summary of each task for the programmer.''', agent=designer)
# task5 = Task(description='''Review every step from the designer, and improve your previous code files accordingly. 
#              Improve the CSS to an Enterprise level (and expand upon the html and js files where neccessary). 
#              Use Flexbox and modern web-design standards. make sure its responsive for phones. Output each new file one at a time.''', agent=coder)
# task6 = Task(description='''Review the code. Think about the next steps towards completing the project (main goal). 
#              Make a detailed summary of each task for the programmer. Have the designer add Bootstrap to improve the design.''', agent=designer)
# task7 = Task(description='''Review every step from the designer, and improve your previous code files accordingly. 
#              Output each new file one at a time.''', agent=coder)
# task8 = Task(description='''Review and test the code. output improved code. Also use the dall-e image generator tool to create images for the website, and input them into the code.''', agent=tester)
# task9 = Task(description='''Think about the design of the website. Review the current state of the code. 
#              Your job is to provide copy for the website to the coder agent.''', agent=designer)
# task10 = Task(description='''Take advice from the designer. Write the css and js files to improve the UI and UX.''', agent=coder)
# task9 = Task(description='''Elaborate ''', agent=coder)
# task4 = Task(description='Add a simple node and express backend server.', agent=coder)
# task5 = Task(description='setup a database with sqlite', agent=coder)
# task6 = Task(description='add database entries for videos, projects, and users.', agent=coder)
task7 = Task(description='write a story arc and develop characters.', agent=producer)
task8 = Task(description='write a comedy tv show script surrounding the characters and process of building the app that youve been handed. Make sure there is banter, struggle, arguements, hilarious jokes in the dialogue. Seperate into 5 scenes, each with at least 10 lines.', agent=producer)
# task9 = Task(description='You are now a salesperson. Sell the app you made, tell them where to buy it at theaiffice.com. and tell them thanks for watching.', agent=producer)

crew = Crew(
  agents=[researcher, coder, designer, producer],
  tasks=[task1, task3, task4, task5, task7, task8],
  verbose=2,
  process=Process.sequential
)

result = crew.kickoff()