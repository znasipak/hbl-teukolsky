import collocode
from mode import TeukolskyPointParticleModeGenerator, TeukolskyPointParticleModeGridGenerator
from geo import MiniGeo
import numpy as np

orbit = MiniGeo(0.9, 20., 0., 1)

solver = collocode.CollocationODEMultiDomainFixedStepSolver(n=32, chtype=1)
solver_kwargs = {"subdomains": 150, "tol": 1e-13, "spacing": 'arcsinh7'}
teuk_gen = TeukolskyPointParticleModeGenerator(orbit, solver, solver_kwargs)

teuk_grid_gen = TeukolskyPointParticleModeGridGenerator(orbit, solver=solver, solver_kwargs = solver_kwargs)
teuk_grid_gen.optimize(-2, 80, solver=solver)

psi = teuk_grid_gen(-2, [2, 20])