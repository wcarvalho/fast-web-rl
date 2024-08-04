import secrets
from typing import Optional, List

from fastapi import Cookie
from fasthtml.common import Request
from fasthtml.common import *
from starlette.responses import Response
import jax.numpy as jnp
import jax.random
import random


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

    def load_session(self, request: Request):
        print('-'*50)
        print('loaded')
        print(request.session)

        if 'session_id' not in request.session:
            request.session['session_id'] = random.getrandbits(32)
        session_id = request.session['session_id']

        if 'rng_key' not in request.session:
            user_seed = int(session_id)  # Convert hex string to int
            rng_key = jax.random.PRNGKey(user_seed)
            request.session['rng_key'] = rng_key.tolist()
        
        if 'stage_idx' not in request.session:
            request.session['stage_idx'] = 0

        print('updates')
        print(request.session)

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

    def load_stage(self, request: Request):
        stage = self.stages[request.session['stage_idx']]

        return Body(
            Main(
                stage.load_stage(request),
                #ws_send="",
                #hx_ext="ws",
                #ws_connect="/wscon",
            ),
            #Script("""
            #document.addEventListener('DOMContentLoaded', async function () {
            #    me(document).on('keydown', ev => {
            #        if (ev.ctrlKey && shortcuts[ev.key.toLowerCase()]) {
            #            ev.preventDefault()
            #            console.load(ev.key)
            #        }
            #    })
            #})
            #       """),
            Script("""
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.host;
                const wsUrl = `${protocol}//${host}/ws`;

                const socket = new WebSocket(wsUrl);
                console.log('socket created');
                document.addEventListener("keydown", function(event) {
                    console.log(event.key);
                    if (socket.readyState === WebSocket.OPEN) {
                        socket.send(JSON.stringify({key: event.key}));
                    }
                });

            """),
            #Form(Group(Input(), Button("Send", cls="btn btn-primary")),
            #     ws_send="", hx_ext="ws", ws_connect="/wscon"),
            id="content"
        )
