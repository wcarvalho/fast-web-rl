from typing import Union
from base64 import b64encode
import inspect
import jax.numpy as jnp
import jax.random
import numpy as np
from fasthtml.common import Request
from flax import struct
from typing import get_type_hints
import io
from PIL import Image


def new_rng(request: Request):
    rng_key = jnp.array(
        request.session['rng_key'], dtype=jax.numpy.uint32)
    rng_key, rng = jax.random.split(rng_key)
    request.session['rng_key'] = rng_key.tolist()
    return rng

def deserialize(cls: struct.PyTreeNode, data: dict):
    """
    Automatically deserialize data into the given class.
    
    Args:
    cls: The class to deserialize into
    data: The data to deserialize
    
    Returns:
    An instance of cls with deserialized data
    """
    if isinstance(data, (str, int, float, bool)):
        return data

    if isinstance(data, jnp.ndarray):
        return data

    if isinstance(data, list):
        return [deserialize(cls, item) for item in data]

    if cls == jnp.ndarray:
        if isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
          return jnp.array([data[str(i)] for i in range(len(data))])
        else:
            raise NotImplementedError(type(data))

    if isinstance(data, dict):
        hints = get_type_hints(cls)
        kwargs = {}
        for key, value in data.items():
            if key in hints:
                field_type = hints[key]
                if inspect.isclass(field_type) and (
                        issubclass(field_type, struct.PyTreeNode) or hasattr(field_type, '__annotations__')):
                    kwargs[key] = deserialize(field_type, value)

                elif field_type == jnp.ndarray:
                    # Convert dict to list if it's a 1D array
                    if isinstance(value, dict) and all(k.isdigit() for k in value.keys()):
                        value = [value[str(i)] for i in range(len(value))]
                    kwargs[key] = jnp.array(value)
                else:
                    kwargs[key] = value
            else:
                kwargs[key] = value
        return cls(**kwargs)

    raise ValueError(f"Unable to deserialize {data} into {cls}")


def encode_image(image: np.ndarray):
    buffer = io.BytesIO()
    Image.fromarray(image.astype('uint8')).save(buffer, format="JPEG")
    encoded_image = b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64,' + encoded_image













def default_timestep_output(
        stage,
        timestep):
    state_image = stage.render_fn(timestep, stage.env_params)
    processed_image = encode_image(state_image)
    return processed_image


def default_evaluate_success(timestep):
    return int(timestep.reward > .8)


class JaxWebEnv:

    def __init__(
            self,
            env,
            #keyparser = None,
            #timestep_output_fn=None,
            #evaluate_success_fn=None,
            #task_name_fn=None
    ):
        self.env = env
        #self.keyparser = keyparser
        #timestep = None

        #if timestep_output_fn is None:
        #    timestep_output_fn = default_timestep_output
        #self.timestep_output = timestep_output_fn

        #if evaluate_success_fn is None:
        #    evaluate_success_fn = default_evaluate_success
        #self.evaluate_success = evaluate_success_fn

        #if task_name_fn is None:
        #    def task_name_fn(*args, **kwargs): 'task'
        #self.task_name = task_name_fn

        def next_step(rng, timestep, action, env_params):
          return env.step(
              rng, timestep, action, env_params)

        def next_steps(rng, timestep, env_params):
            actions = jnp.arange(env.num_actions)
            rngs = jax.random.split(rng, env.num_actions)

            # vmap over rngs and actions. re-use timestep
            timesteps = jax.vmap(
                next_step, in_axes=(0, None, 0, None), out_axes=0
            )(rngs, timestep, actions, env_params)
            return timesteps

        self.reset = jax.jit(self.env.reset)
        self.next_steps = jax.jit(next_steps)

    #def reset(self, rng, env_params):
    #    timestep = self.env.reset(rng, env_params)

    #    return timestep

    def step(self, action_key, env_params, rng):
        action = self.keyparser.action(action_key)
        timestep = self.env.step(
            rng, timestep, action, env_params)

        return timestep
