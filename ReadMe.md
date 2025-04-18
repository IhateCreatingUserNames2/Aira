# AIRA Hub - Tool and Agent Management Platform

This README provides a comprehensive guide for setting up, connecting to, and using the AIRA Hub, a unified platform for managing both Model Context Protocol (MCP) tools and Agent-to-Agent (A2A) skills.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup Instructions](#setup-instructions)
3. [Authentication](#authentication)
   - [OAuth 2.1 Registration](#oauth-21-registration)
   - [Obtaining Access Tokens](#obtaining-access-tokens)
   - [Token Refresh](#token-refresh)
4. [Connecting Your Agent](#connecting-your-agent)
   - [Agent Registration](#agent-registration)
   - [Status Maintenance](#status-maintenance)
5. [Offering Tools/Skills](#offering-toolsskills)
   - [MCP Tools](#mcp-tools)
   - [A2A Skills](#a2a-skills)
6. [Invoking Tools/Skills](#invoking-toolsskills)
   - [Invoking MCP Tools](#invoking-mcp-tools)
   - [Invoking A2A Skills](#invoking-a2a-skills)
7. [Managing Your Resources](#managing-your-resources)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

## Introduction

AIRA Hub is a central platform for registering, managing, and invoking MCP tools and A2A agent skills. It provides:

- Unified registry for both MCP and A2A capable agents
- OAuth 2.1 authentication
- SSE-based connection monitoring
- Tool/skill invocation proxying
- Agent status tracking

## Setup Instructions

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn httpx pydantic jwt python-jose
   ```
3. Start the server:
   ```bash
   python AiraHub.py --host 0.0.0.0 --port 8015
   ```
4. Access the API documentation at `http://localhost:8015/docs`

## Authentication

The AIRA Hub uses OAuth 2.1 for authentication, supporting both authorization code flow and client credentials flow.

### OAuth 2.1 Registration

1. Register an OAuth client:
   ```bash
   curl -X POST "http://localhost:8015/register" \
     -H "Authorization: Bearer {your_access_token}" \
     -H "Content-Type: application/json" \
     -d '{"redirect_uris": ["http://localhost:3000/callback"], "client_name": "My App"}'
   ```

2. Save the returned `client_id` and `client_secret`

### Obtaining Access Tokens

#### Password Grant (Development Only)

```bash
curl -X POST "http://localhost:8015/token" \
  -d "username=agent1&password=password123&grant_type=password" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

#### Authorization Code Flow (Recommended)

1. Redirect user to:
   ```
   http://localhost:8015/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=profile%20agent:read%20agent:write
   ```

2. After approval, exchange code for tokens:
   ```bash
   curl -X POST "http://localhost:8015/token" \
     -d "grant_type=authorization_code&code={code}&redirect_uri={redirect_uri}&client_id={client_id}&client_secret={client_secret}" \
     -H "Content-Type: application/x-www-form-urlencoded"
   ```

### Token Refresh

Refresh tokens are provided as HTTP-only cookies. To refresh:

```bash
curl -X POST "http://localhost:8015/refresh-token" \
  -H "Cookie: refresh_token={refresh_token}"
```

## Connecting Your Agent

### Agent Registration

There are two ways to register your agent with AIRA Hub:

#### 1. SSE Connection (Recommended)

Establish a Server-Sent Events (SSE) connection for automatic heartbeat management:

```bash
curl -N "http://localhost:8015/connect/stream?agent_url=http://localhost:10000&name=MyAgent&aira_capabilities=mcp,a2a" \
  -H "Authorization: Bearer {your_access_token}"
```

The server will respond with an SSE stream containing heartbeats and connection status.

#### 2. Update Agent Metadata

After initial connection, update your agent's metadata:

```bash
curl -X POST "http://localhost:8015/connect/stream/init" \
  -H "Authorization: Bearer {your_access_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://localhost:10000",
    "mcp_capabilities": {
      "tools": {},
      "resources": {"subscribe": true}
    },
    "mcp_tools": [
      {
        "name": "generate_text",
        "description": "Generate text based on a prompt",
        "inputSchema": {
          "type": "object",
          "properties": {
            "prompt": {"type": "string"}
          },
          "required": ["prompt"]
        }
      }
    ],
    "tags": ["text", "generation"]
  }'
```

### Status Maintenance

The SSE connection automatically manages your agent's status. If you disconnect, your agent will be marked as offline after 30 seconds of inactivity.

## Offering Tools/Skills

### MCP Tools

To expose MCP tools:

1. Include the tools in your agent registration or update
2. Implement the MCP tool endpoint in your agent
3. Ensure each tool has:
   - `name`: Unique identifier
   - `description`: Human-readable description
   - `inputSchema`: JSON Schema for arguments
   - Optional `annotations`: Additional tool metadata

Example tool definition:

```json
{
  "name": "weather_lookup",
  "description": "Look up weather information for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City or zip code"},
      "units": {"type": "string", "enum": ["metric", "imperial"]}
    },
    "required": ["location"]
  }
}
```

### A2A Skills

To expose A2A skills:

1. Include the skills in your agent registration or update
2. Implement the A2A task handler in your agent
3. Ensure each skill has:
   - `id`: Unique identifier
   - `name`: Human-readable name
   - `description`: Human-readable description
   - Optional `tags`: Categorization labels
   - Optional `examples`: Example prompts

Example skill definition:

```json
{
  "id": "summarize_text",
  "name": "Text Summarization",
  "description": "Summarize a long text into key points",
  "tags": ["text", "summarization"],
  "examples": ["Summarize this article", "Give me the key points from this text"]
}
```

## Invoking Tools/Skills

### Invoking MCP Tools

To invoke an MCP tool:

```bash
curl -X POST "http://localhost:8015/mcp/invoke-tool" \
  -H "Authorization: Bearer {your_access_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_url": "http://localhost:10000",
    "tool_name": "weather_lookup",
    "arguments": {
      "location": "New York",
      "units": "metric"
    }
  }'
```

Response:

```json
{
  "success": true,
  "message": "Tool 'weather_lookup' executed successfully",
  "content": [
    {
      "type": "text",
      "text": "Current weather in New York: 22°C, Partly Cloudy"
    }
  ]
}
```

### Invoking A2A Skills

To invoke an A2A skill:

```bash
curl -X POST "http://localhost:8015/a2a/invoke-skill" \
  -H "Authorization: Bearer {your_access_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_url": "http://localhost:10000",
    "tool_name": "summarize_text",
    "arguments": {
      "prompt": "Summarize the following text: Lorem ipsum dolor sit amet..."
    }
  }'
```

Response:

```json
{
  "success": true,
  "message": "Skill 'summarize_text' executed successfully",
  "content": [
    {
      "type": "text",
      "text": "The text discusses the importance of clear communication..."
    }
  ]
}
```

### Checking Tool Call Status

To check the status of a previous tool call:

```bash
curl "http://localhost:8015/tools/calls/{call_id}" \
  -H "Authorization: Bearer {your_access_token}"
```

## Managing Your Resources

### Listing Your Agents

```bash
curl "http://localhost:8015/my/agents" \
  -H "Authorization: Bearer {your_access_token}"
```

### Listing Your OAuth Clients

```bash
curl "http://localhost:8015/my/oauth-clients" \
  -H "Authorization: Bearer {your_access_token}"
```

### Viewing Available Tools and Skills

```bash
# List all MCP tools
curl "http://localhost:8015/mcp/tools" \
  -H "Authorization: Bearer {your_access_token}"

# List all A2A skills
curl "http://localhost:8015/a2a/skills" \
  -H "Authorization: Bearer {your_access_token}"
```

## API Reference

Full API documentation is available at `http://localhost:8015/docs` when the server is running.

Key endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/connect/stream` | GET | Establish SSE connection for agent |
| `/connect/stream/init` | POST | Update agent metadata |
| `/status` | GET | Get status of all agents |
| `/mcp/tools` | GET | List all MCP tools |
| `/a2a/skills` | GET | List all A2A skills |
| `/mcp/invoke-tool` | POST | Invoke an MCP tool |
| `/a2a/invoke-skill` | POST | Invoke an A2A skill |
| `/my/agents` | GET | List your agents |
| `/my/oauth-clients` | GET | List your OAuth clients |
| `/tools/calls/{call_id}` | GET | Check tool call status |

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Check token expiration (use `/refresh-token`)
   - Verify permissions for the requested operation

2. **Agent Connection Issues**
   - Ensure your agent URL is accessible
   - Check that SSE connection is maintained
   - Verify that your agent implements the required endpoints

3. **Tool Invocation Failures**
   - Verify tool exists on the agent
   - Check argument schema matches requirements
   - Ensure agent is online

### Logs

For detailed logs, run the AIRA Hub with increased logging:

```bash
python AiraHub.py --host 0.0.0.0 --port 8015 --log-level debug
```

---

Example use

![image](https://github.com/user-attachments/assets/bf47dfc5-272d-47e8-9e57-8f849f0e3ee9)
