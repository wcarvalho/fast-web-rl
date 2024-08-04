import secrets
from typing import Optional, List

from fastapi import Cookie
from fasthtml.common import Request
from fasthtml.common import Body, Main
from starlette.responses import Response
import jax.numpy as jnp
import jax.random
import random

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from starlette.endpoints import WebSocketEndpoint

from fastwebrl.stages import Stage


class ExperimentManager:
    def __init__(
            self,
            stages: List[Stage],
            #cookie_name: str = "session_id",
            #secret_key: str = "your-secret-key-here"
            ):
        #self.cookie_name = cookie_name
        #self.secret_key = secret_key
        self.stages = stages

    def sesson_from_cookies(self, cookies: dict):
        session_id = cookies.get("session_id", random.getrandbits(32))
        rng_key = cookies.get("rng_key", None)
        if rng_key is None:
            user_seed = int(session_id)  # Convert hex string to int
            rng_key = jax.random.PRNGKey(user_seed)
        stage_idx = cookies.get("stage_idx", 0)

        return {
            "session_id": session_id,
            "stage_idx": stage_idx,
            "rng_key": rng_key
        }

    def load_session(self, request: Request, response: Response):
        print('-'*50)
        print('loaded')

        session = self.sesson_from_cookies(request.cookies)
        response.set_cookie(
            key="session_id", 
            value=session['session_id'],
            httponly=True, samesite="strict")
        response.set_cookie(
            key="stage_idx", 
            value=session['stage_idx'],
            httponly=True, samesite="strict")
        response.set_cookie(
            key="rng_key", 
            value=session['rng_key'].tolist(),
            httponly=True, samesite="strict")

        return session
        ##print(request.session)

        #if 'session_id' not in request.session:
        #    request.session['session_id'] = random.getrandbits(32)
        #session_id = request.session['session_id']

        #if 'rng_key' not in request.session:
        #    rng_key = jax.random.PRNGKey(user_seed)
        #    request.session['rng_key'] = rng_key.tolist()
        
        #if 'stage_idx' not in request.session:
        #    request.session['stage_idx'] = 0

        #print('updates')
        #print(request.session)

    def get_stage_idx(self, request: Request):
        return request.session['stage_idx']

    def decrement_stage(self, request: Request):
        request.session['stage_idx'] = max(
            0,
            request.session['stage_idx'] - 1)
        print('decrement_stage')
        print(request.session)

    def increment_stage(self, request: Request):
        request.session['stage_idx'] = min(
            request.session['stage_idx'] + 1,
            len(self.stages))
        print('increment_stage')
        print(request.session)

    def load_stage(self, session):
        stage = self.stages[request.session['stage_idx']]

        return Body(
            Main(stage.load_stage(request)),
            id="content"
        )

