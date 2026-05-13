with open("agent_factory.py", "r") as f:
    content = f.read()
if "def act(" not in content.split("class DynamicAgent(")[1].split("class AgentFactory(")[0]:
    print("No act method in DynamicAgent")
