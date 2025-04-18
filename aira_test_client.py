import requests

AIRA_HUB = "https://aira-fl8f.onrender.com"
USERNAME = "admin"
PASSWORD = "password123"

def get_token():
    print("🔐 Autenticando...")
    resp = requests.post(
        f"{AIRA_HUB}/token",
        data={"username": USERNAME, "password": PASSWORD},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    if resp.status_code != 200:
        print("❌ Falha ao autenticar:", resp.text)
        return None

    token = resp.json()["access_token"]
    print("✅ Token recebido!")
    return token

def list_agents(token):
    print("🌐 Listando agentes registrados...")
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{AIRA_HUB}/status", headers=headers)

    if resp.status_code != 200:
        print("❌ Erro ao listar agentes:", resp.text)
        return []

    data = resp.json()
    for ag in data.get("agents", []):
        print(f"🧠 {ag['name']} ({ag['status']}) - {ag['url']}")
    return data.get("agents", [])

def find_dummy(agents):
    for ag in agents:
        if ag["name"] == "DummyTestAgent":
            print("✅ DummyTestAgent encontrado!")
            return ag
    print("❌ DummyTestAgent não encontrado.")
    return None

def main():
    token = get_token()
    if not token:
        return

    agents = list_agents(token)
    dummy = find_dummy(agents)

    if dummy:
        print("🚀 Tudo pronto. Dummy está conectado e visível!")
        # Aqui você pode adicionar chamadas como invoke_tool()
    else:
        print("😭 Nenhum agente dummy foi encontrado. Está rodando o dummy_agent_sse.py?")

if __name__ == "__main__":
    main()
