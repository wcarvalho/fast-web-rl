import secrets
from typing import Optional, List

from fastapi import Cookie
from fasthtml.common import Request
from fasthtml.common import Body, Main
from starlette.responses import Response
import jax.numpy as jnp
import jax.random
import json
import random

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from starlette.endpoints import WebSocketEndpoint
from starlette.datastructures import MutableHeaders


from fastwebrl.managers import ExperimentManager

class WebSocketHandler(WebSocketEndpoint):
    encoding = "json"

    def __init__(self, scope, receive, send, manager: ExperimentManager):
        super().__init__(scope, receive, send)
        self.manager = manager
        self.session = {}
        self.headers = MutableHeaders(scope=scope)

    async def on_connect(self, websocket: WebSocket) -> None:
        await websocket.accept()

        cookies = self.scope.get('headers', [])
        cookie_dict = dict(cookie.split('=') for cookie in cookies if cookie[0] == b'cookie')

        self.session = self.manager.sesson_from_cookies(cookie_dict)
        self.update_cookies()

    def update_cookies(self):
        self.headers.append(
            'Set-Cookie',
            f"session_id={self.session['session_id']}; Path=/; HttpOnly; SameSite=Strict")
        self.headers.append(
            'Set-Cookie',
            f"stage_idx={self.session['stage_idx']}; Path=/; HttpOnly; SameSite=Strict")
        self.headers.append(
            'Set-Cookie',
            f"rng_key={self.session['rng_key'].tolist()}; Path=/; HttpOnly; SameSite=Strict")

    async def on_receive(self, websocket: WebSocket, data: dict) -> None:
        mock_request = Request(scope=websocket.scope)
        key = data.get('key')
        print(f"Received key: {key}")
        import ipdb
        ipdb.set_trace()
        # await websocket.send_json({"type": "stage_content", "content": stage_content.dict()})

    async def on_disconnect(self, websocket: WebSocket, close_code: int) -> None:
        print("WebSocket disconnected")
