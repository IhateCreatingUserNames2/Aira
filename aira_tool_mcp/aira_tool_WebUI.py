from flask import Flask, request, jsonify, render_template_string
import aiohttp
import asyncio
import requests
import json
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AIRA Hub - Remote MCP Server Registration</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            border-bottom: 1px solid #ccc;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"], input[type="url"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #f9f9f9;
        }
        .success {
            color: green;
        }
        .error {
            color: red;
        }
        .server-list {
            margin-top: 30px;
        }
        .server-item {
            border: 1px solid #eee;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f5f5f5;
        }
        .server-name {
            font-weight: bold;
        }
        .server-url {
            color: #666;
            font-size: 0.9em;
        }
        .tool-list {
            margin-top: 10px;
        }
        .tool-item {
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 5px 10px;
            margin: 5px 0;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>AIRA Hub - Remote MCP Server Registration</h1>

    <div class="form-group">
        <label for="hub-url">AIRA Hub URL:</label>
        <input type="url" id="hub-url" value="https://aira-fl8f.onrender.com" placeholder="Enter AIRA Hub URL">
    </div>

    <div class="form-group">
        <label for="server-url">MCP Server URL:</label>
        <input type="url" id="server-url" placeholder="Enter Remote MCP Server URL">
    </div>

    <button onclick="registerServer()">Register MCP Server</button>

    <div id="result" class="result" style="display: none;"></div>

    <div class="server-list">
        <h2>Registered Servers</h2>
        <button onclick="refreshServers()">Refresh Server List</button>
        <div id="servers-container"></div>
    </div>

    <script>
        // Load servers on page load
        document.addEventListener('DOMContentLoaded', function() {
            refreshServers();
        });

        function registerServer() {
            const hubUrl = document.getElementById('hub-url').value;
            const serverUrl = document.getElementById('server-url').value;
            const resultDiv = document.getElementById('result');

            if (!serverUrl) {
                resultDiv.innerHTML = '<span class="error">Please enter a server URL</span>';
                resultDiv.style.display = 'block';
                return;
            }

            resultDiv.innerHTML = 'Registering server...';
            resultDiv.style.display = 'block';

            fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    hub_url: hubUrl,
                    server_url: serverUrl
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    resultDiv.innerHTML = `<span class="success">${data.message}</span>`;
                    // Clear server URL and refresh list
                    document.getElementById('server-url').value = '';
                    refreshServers();
                } else {
                    resultDiv.innerHTML = `<span class="error">${data.message}</span>`;
                }
            })
            .catch(error => {
                resultDiv.innerHTML = `<span class="error">Error: ${error.message}</span>`;
            });
        }

        function refreshServers() {
            const hubUrl = document.getElementById('hub-url').value;
            const serversContainer = document.getElementById('servers-container');

            serversContainer.innerHTML = 'Loading servers...';

            fetch('/api/agents?hub_url=' + encodeURIComponent(hubUrl))
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (data.agents.length === 0) {
                        serversContainer.innerHTML = '<p>No servers registered yet.</p>';
                        return;
                    }

                    let html = '';
                    data.agents.forEach(agent => {
                        if (agent.tags && agent.tags.includes('mcp')) {
                            html += `
                                <div class="server-item">
                                    <div class="server-name">${agent.name}</div>
                                    <div class="server-url">${agent.url}</div>
                                    <div class="tool-list">
                                        <strong>Tools:</strong>
                                        ${agent.skills.length ? '' : '<p>No tools available</p>'}
                                    </div>
                            `;

                            if (agent.skills.length) {
                                html += '<ul>';
                                agent.skills.forEach(skill => {
                                    html += `<li class="tool-item">${skill.name} - ${skill.description || ''}</li>`;
                                });
                                html += '</ul>';
                            }

                            html += '</div>';
                        }
                    });

                    serversContainer.innerHTML = html || '<p>No MCP servers found.</p>';
                } else {
                    serversContainer.innerHTML = `<p class="error">${data.message}</p>`;
                }
            })
            .catch(error => {
                serversContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/register', methods=['POST'])
def register_server():
    data = request.json
    hub_url = data.get('hub_url', 'https://aira-fl8f.onrender.com')
    server_url = data.get('server_url')

    if not server_url:
        return jsonify({"success": False, "message": "Server URL is required"})

    try:
        # First, discover MCP server capabilities
        try:
            # Try to get server info
            server_info_response = requests.post(
                f"{server_url}/initialize",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                timeout=5
            )

            if server_info_response.status_code == 200:
                server_info = server_info_response.json()
                server_name = server_info.get("result", {}).get("server_name", "Unknown MCP Server")
            else:
                server_name = os.path.basename(server_url)

            # Try to list tools
            tools_response = requests.post(
                f"{server_url}/list_tools",
                json={"jsonrpc": "2.0", "id": 2, "method": "list_tools"},
                timeout=5
            )

            if tools_response.status_code == 200:
                tools = tools_response.json().get("result", [])
            else:
                tools = []

        except requests.RequestException as e:
            return jsonify({"success": False, "message": f"Error connecting to MCP server: {str(e)}"})

            # Convert MCP tools to A2A skills
        skills = []
        for tool in tools:
            skill_id = f"tool-{tool.get('name', 'unknown')}"
            skill_name = tool.get("name", "Unknown Tool")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})

            skills.append({
                "id": skill_id,
                "name": skill_name,
                "description": description,
                "tags": ["mcp", "tool"],
                "parameters": parameters
            })

        # Create registration payload
        payload = {
            "url": server_url,
            "name": f"{server_name}",
            "description": f"Remote MCP Server exposed through AIRA Hub",
            "skills": skills,
            "shared_resources": [],
            "aira_capabilities": ["mcp", "a2a"],
            "auth": {},
            "tags": ["mcp", "remote-server"]
        }

        # Register with AIRA hub
        hub_response = requests.post(f"{hub_url}/register", json=payload)

        if hub_response.status_code == 201:
            return jsonify({
                "success": True,
                "message": f"Successfully registered MCP server {server_name} with {len(skills)} tools"
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to register with AIRA Hub: {hub_response.text}"
            })

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


@app.route('/api/agents')
def list_agents():
    hub_url = request.args.get('hub_url', 'https://aira-fl8f.onrender.com')

    try:
        response = requests.get(f"{hub_url}/agents")

        if response.status_code == 200:
            agents = response.json()
            return jsonify({"success": True, "agents": agents})
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to retrieve agents: {response.text}"
            })

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)