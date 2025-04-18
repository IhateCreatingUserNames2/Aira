import asyncio
import aiohttp
import json

AIRA_HUB = "https://aira-fl8f.onrender.com"
USERNAME = "admin"
PASSWORD = "password123"

AGENT_NAME = "DummyTestAgent"
AGENT_URL = "http://localhost:9999"
CAPABILITIES = "mcp,a2a"

async def main():
    async with aiohttp.ClientSession() as session:
        # Etapa 1: Autenticação
        token_resp = await session.post(
            f"{AIRA_HUB}/token",
            data={
                "username": USERNAME,
                "password": PASSWORD
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if token_resp.status != 200:
            print("❌ Erro ao autenticar")
            print(await token_resp.text())
            return

        token_json = await token_resp.json()
        headers = {
            "Authorization": f"Bearer {token_json['access_token']}"
        }

        # Etapa 2: Conectar SSE
        stream_url = f"{AIRA_HUB}/connect/stream?agent_url={AGENT_URL}&name={AGENT_NAME}&aira_capabilities={CAPABILITIES}"
        async with session.get(stream_url, headers=headers) as resp:
            if resp.status != 200:
                print("❌ Erro ao conectar SSE:", resp.status)
                return

            print("✅ Conectado ao AIRA Hub! Enviando init...")

            # Etapa 3: Enviar init com dados básicos
            init_payload = {
                "url": AGENT_URL,
                "description": "A dummy agent for test purposes",
                "skills": [],
                "shared_resources": [],
                "tags": ["streamed", "test"],
                "category": "test"
            }

            post_resp = await session.post(f"{AIRA_HUB}/connect/stream/init", json=init_payload, headers=headers)

            if post_resp.status == 200:
                print("✅ Init enviado com sucesso!")
            else:
                print("❌ Erro ao enviar init:", await post_resp.text())
                return

            # Etapa 4: Ler SSE
            async for line in resp.content:
                if not line:
                    continue
                decoded = line.decode().strip()
                if decoded.startswith("data: "):
                    event = json.loads(decoded[6:])
                    data = event.get("data", {})
                    print(f"[{data.get('agent')}] {data.get('status')} @ {data.get('ts')}")

if __name__ == "__main__":
    asyncio.run(main())
