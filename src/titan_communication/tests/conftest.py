"""Shared pytest fixtures.

Populated as we go. Expected residents:
    - ``sim_env``: a fresh ``simpy.Environment`` per test
    - ``deterministic_rng``: seeded ``numpy.random.Generator`` for
      channel/shadowing tests
    - ``default_lora_params``: the paper's SF=10 / BW=500 kHz / CR=4/5
      / TX=15 dBm preset
"""
