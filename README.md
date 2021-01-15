# AmongAIs-AI0

To run the agent, simply download the ai0-agent.py file from the Github repository, and run it in a terminal with
>python ai0-agent.py matchname playername
  
The agent will start playing the game as soon as the game is started. You can monitor the actions taken by the AI player by reading the output console.

We also provide a simple “runner” agent, which only runs towards the flag trying to avoid enemies without shooting.
You can run this agent by downloading the ai0-runner.py file and running it with 

>python ai0-runner.py matchname playername

You may be required to install the python-pathfinding library. In that case, you need to run in a terminal

>pip install pathfinding
