
# 🤖 AIRA Hub – Agent Interoperability & Resource Access

**AIRA Hub** is a decentralized interoperability layer that connects AI agents built with different protocols (A2A and MCP). It serves as a registry, discovery service, and live heartbeat monitor for agents, tools, and skills across a distributed AI network.

---

## 🌐 Overview

AIRA enables hybrid AI ecosystems, allowing agents with different capabilities to **discover each other**, **share tools or skills**, and **remain synchronized** via live SSE (Server-Sent Events) updates.

Supported agent types:
- **MCP Agents** – offer tools/resources using the Model Context Protocol
- **A2A Agents** – offer high-level skills for direct use by other agents
- **Hybrid Agents** – expose both tools and skills

---

## 🚀 Getting Started

### 1. **Run AIRA Hub**

Run the AIRA Hub server:

```bash
python AiraHub.py --port 8015
The default database is aira_db.json. Agents register here to announce their skills/tools and sync heartbeat status.

2. Connect an Agent via SSE
Use the provided client: aira_sse_client.py

This script:

Registers the agent with live heartbeat (/connect/stream)

Initializes its metadata (/connect/stream/init)

Keeps connection alive and updates last-seen timestamps

bash
Copy
Edit
python aira_sse_client.py
✅ Example registration includes:

An A2A skill: semantic-recall

An MCP tool: recall

🧠 How AIRA Hub Organizes Agents

Route Prefix	Client View	Filter Logic
/mcp/agents	MCP-only view	Only shows agents with "mcp"
/a2a/agents	A2A-only view	Only shows agents with "a2a"
/hybrid/agents	Full ecosystem view	Includes agents with any capability
/events/stream	SSE agent heartbeat	Streams status for individual agent
🛠 How to Discover and Use Tools & Skills
You can use the script aira_tool_user.py to:

🔍 Discover agents on the network

📡 Call MCP Tools via HTTP POST

🎯 Use A2A Skills (metadata only, execution is up to implementation)

Example Usage:
bash
Copy
Edit
python aira_tool_user.py
Sample output:

yaml
Copy
Edit
🤖 MCP Agent: MemoryAgent
🚀 Calling tool: http://localhost:8094/tools/recall
✅ Tool response: {"result": "Memory retrieved."}

---

🤖 A2A Agent: MemoryAgent
🎯 Using skill: Semantic Recall from agent MemoryAgent
🧠 Skill Description: Retrieves and reformulates memory based on latent intent
🧬 Agent Registration Schema
Agents register with:

json
Copy
Edit
{
  "url": "http://localhost:8094/",
  "name": "MemoryAgent",
  "aira_capabilities": ["mcp", "a2a"],
  "skills": [...],
  "shared_resources": [...],
  "tags": ["memory", "cognition"],
  "status": "online"
}
💓 SSE Heartbeat Monitoring
Agents connected via /connect/stream are auto-tracked and updated every 5s.

You can view system status via:

http
Copy
Edit
GET /status
Response includes uptime, active agents, and heartbeat lag.

📦 File Structure

File	Purpose
AiraHub.py	Main FastAPI Hub for agent registration/discovery
aira_sse_client.py	SSE-enabled agent that connects to AIRA and syncs
aira_tool_user.py	CLI client to discover and invoke tools/skills
📌 Notes
Hybrid clients like Cognisphere should register with both "mcp" and "a2a" capabilities.

Agent status is auto-updated based on heartbeat TTL (default: 5 minutes).

You can extend the system with more endpoints like /mcp/tools, /a2a/skills, or /invoke.

🧭 Final Words
AIRA is designed to make cross-framework AI collaboration simple, extensible, and real-time. Whether you’re building a tool-focused MCP service or a skill-based multi-agent AI, AIRA lets you plug into the same shared network with zero friction.

Happy connecting! ✨

yaml
Copy
Edit

---

Let me know if you want a version in português ou um diagrama da arquitetura!
