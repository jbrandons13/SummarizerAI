import inspect
import diffusers.pipelines.ltx.pipeline_ltx as pl
print("Signature:", inspect.signature(pl.retrieve_timesteps))
print("Source:\n", inspect.getsource(pl.retrieve_timesteps))
