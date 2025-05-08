# AIRA Hub - Tool and Agent Management Platform



AIRA HUB HAS A NEW VERSION THAT WORKS ON CLAUDE OR OTHER MCP CLIENTS THAT USE JSON OR MCP REMOTE. 

https://github.com/IhateCreatingUserNames2/AiraHub2/tree/main

Test demo on: 
https://airahub2.onrender.com/

CLAUDE JSON CONFIG: 
```
{ 
  "mcpServers": { 
    "aira-hub": { 
      "command": "npx", 
      "args": [ 
        "mcp-remote",
        "https://airahub2.onrender.com/mcp/stream"
      ]
    }
  } 
}
```

## Overview
AIRA Hub is a FastAPI-based platform for managing MCP (Model Context Protocol) tools and A2A (Agent-to-Agent) skills with OAuth 2.1 authentication. This document provides setup instructions, usage examples, and API reference for the system.

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Authentication](#authentication)
3. [Agent Connection](#agent-connection)
4. [Tool/Skill Management](#toolskill-management)
5. [API Reference](#api-reference)
6. [Example Clients](#example-clients)
7. [Troubleshooting](#troubleshooting)

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  pip install fastapi uvicorn httpx pydantic jwt python-jose aiohttp sse-starlette
  ```

### Running the Server
```bash
python AiraHub.py --host 0.0.0.0 --port 8015
```

The API documentation will be available at `http://localhost:8015/docs`

## Authentication

### Default Users
The system comes with these pre-configured users:
- **admin** / password123 (admin privileges)
- **agent1** / password123 (agent privileges)

### Obtaining Tokens
```bash
curl -X POST "http://localhost:8015/token" \
  -d "username=admin&password=password123" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

Response:
```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_at": 1234567890
}
```

### Token Refresh
Refresh tokens are automatically handled via HTTP-only cookies. To manually refresh:
```bash
curl -X POST "http://localhost:8015/refresh-token" \
  -H "Cookie: refresh_token=your_refresh_token"
```

## Agent Connection

### Establishing SSE Connection
```python
# Example from dummy_agent_sse.py
async with session.get(
    f"{AIRA_HUB}/connect/stream?agent_url={AGENT_URL}&name={AGENT_NAME}&aira_capabilities={CAPABILITIES}",
    headers={"Authorization": f"Bearer {token}"}
) as resp:
    # Handle SSE events
```

### Updating Agent Metadata
```python
init_payload = {
    "url": AGENT_URL,
    "description": "Agent description",
    "mcp_tools": [{
        "name": "sample_tool",
        "inputSchema": {"type": "object"}
    }],
    "tags": ["test"]
}

await session.post(f"{AIRA_HUB}/connect/stream/init", json=init_payload)
```

## Tool/Skill Management

### Listing Available Tools
```python
# From aira_test_client.py
resp = requests.get(f"{AIRA_HUB}/mcp/tools", headers=headers)
```

### Invoking an MCP Tool
```python
tool_request = {
    "agent_url": "http://agent-url",
    "tool_name": "tool_name",
    "arguments": {"param": "value"}
}

resp = requests.post(f"{AIRA_HUB}/mcp/invoke-tool", json=tool_request)
```

### Checking Tool Call Status
```python
resp = requests.get(f"{AIRA_HUB}/tools/calls/{call_id}")
```

## API Reference

### Key Endpoints

| Endpoint | Method | Description | Required Role |
|----------|--------|-------------|---------------|
| `/token` | POST | Get access token | - |
| `/connect/stream` | GET | Establish SSE connection | agent/admin |
| `/status` | GET | Get system status | user |
| `/mcp/tools` | GET | List MCP tools | user |
| `/a2a/skills` | GET | List A2A skills | user |
| `/mcp/invoke-tool` | POST | Invoke MCP tool | user |
| `/my/agents` | GET | List user's agents | user |

## Example Clients

Two example clients are provided:

1. **dummy_agent_sse.py** - Demonstrates how to connect an agent via SSE
2. **aira_test_client.py** - Shows how to interact with the hub as a client

To run them:
```bash
python dummy_agent_sse.py
python aira_test_client.py
```

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify username/password
   - Check token expiration

2. **SSE Connection Drops**
   - Ensure your agent maintains the connection
   - Implement reconnection logic

3. **Tool Invocation Errors**
   - Verify the agent is online
   - Check tool name and parameters match the schema

### Logging
For debugging, run the server with:
```bash
python AiraHub.py --host 0.0.0.0 --port 8015 --log-level debug
```

---

This documentation reflects the actual implementation in the provided code files. For the most up-to-date information, always refer to the interactive API docs at `/docs` when the server is running.


### Example 

![image](https://github.com/user-attachments/assets/34cb9f56-c0a3-45bb-b983-e5d05e48e5c0)
