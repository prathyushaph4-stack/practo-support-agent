"""
test_websocket_client.py
Practo Capstone - Task 11: WebSocket test client
----------------------------------------------------
Connects to /ws/chat, sends 2 messages, then disconnects abruptly to prove
the server handles WebSocketDisconnect gracefully (stays running for
other clients).
"""

import asyncio
import websockets
import json


async def run_chat_session():
    uri = "ws://127.0.0.1:8000/ws/chat"
    async with websockets.connect(uri) as ws:
        connected_msg = await ws.recv()
        print(f"[CLIENT] Received: {connected_msg}")

        print("\n[CLIENT] Sending question 1...")
        await ws.send(json.dumps({"question": "What is the cancellation policy?"}))
        response1 = await ws.recv()
        print(f"[CLIENT] Received: {response1}")

        print("\n[CLIENT] Sending question 2...")
        await ws.send(json.dumps({"question": "What is the status of appointment APT0011?"}))
        response2 = await ws.recv()
        print(f"[CLIENT] Received: {response2}")

        print("\n[CLIENT] Disconnecting abruptly (simulating a dropped connection)...")
        # Exiting the `async with` block here closes the connection --
        # the server should log the disconnect and keep running.


if __name__ == "__main__":
    asyncio.run(run_chat_session())