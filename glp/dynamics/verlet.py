from glp.utils import cast

from .utils import *


def verlet(system, velocities, masses, get_calculator, dt=1.0):
    dt = cast(dt)

    calculator, calc_state = get_calculator(system)

    to_V = lambda P: to_velocities(P, masses)
    to_P = lambda V: to_momenta(V, masses)
    to_S = lambda point: to_system(point, system)

    calculate = lambda point, state: calculator.calculate(
        to_S(point),
        state,
        velocities=to_V(point.P),
        masses=masses,
    )
    point = Point(system.R, to_P(velocities))

    results, calc_state = calculate(point, calc_state)

    md_state = MDState(point, results, calc_state, calc_state.overflow)

    def step_fn(md_state, ignored):
        P_halfdt = md_state.point.P + cast(0.5) * dt * md_state.results["forces"]
        R = md_state.point.R + dt * to_V(P_halfdt)

        point = update(md_state.point, R=R)
        results, calc_state = calculate(point, md_state.calc_state)

        P = P_halfdt + cast(0.5) * dt * results["forces"]
        point = update(point, P=P)

        md_state = MDState(point, results, calc_state, calc_state.overflow)

        return md_state, (point, results, calc_state.overflow)

    def update_fn(point):
        return verlet(to_S(point), to_V(point.P), masses, get_calculator, float(dt))

    return Dynamics(step_fn, update_fn), md_state
