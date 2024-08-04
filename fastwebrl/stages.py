from typing import Any, Callable

from fastwebrl.jax import new_rng, encode_image
from flax import struct
from fastcore.all import patch
from fasthtml.common import *

import jax
import jax.numpy as jnp

@struct.dataclass
class StageState:
    finished: bool = False

@struct.dataclass
class Stage:
    name: str = 'stage'
    def load_stage(self, request: Request): pass

@struct.dataclass
class ConsentStage(Stage):
    name: str = 'Experiment Consent'
    instruction: str = 'Instructions.'
    checkbox: str = 'I consent to participate.'

    def load_stage(self, request: Request):
        del request
        consent_form = Div(
            Form(
                Checkbox(
                    id="consent",
                    name="consent",
                    label=self.checkbox),
                Button("Proceed to Experiment", id="proceed-btn", disabled=True),
                hx_post="/experiment",
                hx_trigger="click from:#proceed-btn",
                hx_target='#content',
                hx_swap="innerHTML"
            ),
            hx_ext='ws', ws_connect='/ws'
        )

        script = Script("""
            document.getElementById('consent').addEventListener('change', function() {
                document.getElementById('proceed-btn').disabled = !this.checked;
            });
            """)
        print('loading consent page')
        return Div(
            H1("Consent Form"),
            H3(self.instruction),
            Div(consent_form, cls='container'),
            script,
            )


@struct.dataclass
class EnvStageState(StageState):
    env_timestep: struct.PyTreeNode = None
    nsteps: int = 0
    nepisodes: int = 0
    nsuccesses: int = 0

@struct.dataclass
class EnvStage(Stage):
    instruction: str = 'instruction'
    web_env: Any = None
    env_params: struct.PyTreeNode = None
    reset_env: bool = True
    render_fn: Callable = None
    task_desc_fn: Callable = None
    #next_steps: Callable = field(init=False, default=None)

    #def __post_init__(self):
    #    assert self.env is not None

    #    def next_step(rng, timestep, action):
    #      return self.env.step(
    #          rng, timestep, action, self.env_params)

    #    def next_steps(rng, timestep):
    #        actions = jnp.arange(self.env.num_actions)
    #        rngs = jax.random.split(rng, self.env.num_actions)

    #        # vmap over rngs and actions. re-use timestep
    #        timesteps = jax.vmap(
    #            next_step, in_axes=(0, None, 0), out_axes=0
    #            )(rngs, timestep, actions)
    #        return timesteps

    #    self.next_steps = jax.jit(next_step)

    #def new_state(self, env_timestep): 
    #    return EnvStageState(env_timestep=env_timestep)

    def restart_env(self, request):
        rng = new_rng(request)
        timestep = self.web_env.reset(rng, self.env_params)
        #stage_state = self.new_state(timestep)
        image = self.render_fn(timestep)

        task_desc = self.task_desc_fn(timestep)
        print('loading env page')
        return Div(
            H1(self.name),
            H2(self.instruction),
            task_desc,
            Div(
              Img(src=encode_image(image), id='stateImage'),
              id='stateImageContainer',
            ),
        )

    def load_stage(self, request: Request):
        if 'env_stage_state' in request.session:
            import ipdb; ipdb.set_trace()
        else:
            return self.restart_env(request)
