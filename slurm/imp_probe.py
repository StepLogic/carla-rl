import sys
print("argv0 dir on path:", sys.path[0])
import rlib_integration as r
print("rlib_integration ->", getattr(r, "__file__", None), getattr(r, "__path__", None))
from rlib_integration.carla_env import CarlaEnv  # noqa: F401
print("CarlaEnv OK")
