import requests

AIRA_HUB = "http://localhost:8015"

def discover_mcp_agents():
    url = f"{AIRA_HUB}/mcp/agents"
    response = requests.get(url)
    agents = response.json()
    print(f"🔍 Found {len(agents)} MCP agent(s).")

    for agent in agents:
        tools = [
            r for r in agent["shared_resources"]
            if r["type"] == "mcp_tool"
        ]
        if tools:
            return agent, tools[0]  # return first agent + tool

    return None, None


def call_tool(tool):
    try:
        endpoint = tool["uri"]
        print(f"🚀 Calling tool: {endpoint}")

        response = requests.post(endpoint, json={"input": "Hello AIRA!"})
        if response.ok:
            print(f"✅ Tool response: {response.json()}")
        else:
            print(f"❌ Tool failed: {response.status_code} | {response.text}")
    except Exception as e:
        print(f"⚠️ Error calling tool: {e}")

def discover_a2a_agents():
    url = f"{AIRA_HUB}/a2a/agents"
    response = requests.get(url)
    agents = response.json()
    print(f"🔍 Found {len(agents)} A2A agent(s).")

    for agent in agents:
        skills = agent.get("skills", [])
        if skills:
            return agent, skills[0]  # return first agent + skill

    return None, None


def use_a2a_skill(agent, skill):
    print(f"🎯 Using skill: {skill['name']} from agent {agent['name']}")
    # Simular uso da skill — isso depende da implementação A2A real
    # Aqui só mostramos que foi encontrada
    print(f"🧠 Skill Description: {skill['description']}")
    print(f"📎 Tags: {skill.get('tags', [])}")



if __name__ == "__main__":
    agent, tool = discover_mcp_agents()
    if agent and tool:
        print(f"🤖 MCP Agent: {agent['name']}")
        call_tool(tool)
    else:
        print("❌ No MCP agents with tools found.")

    print("\n---\n")

    agent, skill = discover_a2a_agents()
    if agent and skill:
        print(f"🤖 A2A Agent: {agent['name']}")
        use_a2a_skill(agent, skill)
    else:
        print("❌ No A2A agents with skills found.")

