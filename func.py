import numpy as np
import time

# ======= Layered directed-graph construction =======
import numpy as np

def make_ContinuousVar_candidates(xmin, xmax, xinit, xend, add_x_set,
                      ru_min, ru_max, rd_min, rd_max, T, eps=1e-10):
    """Construct candidate values of x from anchors and upward/downward step endpoints."""
    print("开始构建储能机组的 x 候选集...")

    digits = 10

    step_up = {round(ru_min, digits), round(ru_max, digits)}
    step_down = {round(rd_min, digits), round(rd_max, digits)}

    steps_forward = step_up | {-step for step in step_down}
    steps_backward = {-step for step in step_up} | step_down

    add_x_set = {
        round(x, digits)
        for x in add_x_set
        if x is not None and xmin + eps <= x <= xmax - eps
    }

    x_base_forward = set() if xinit is None else {round(xinit, digits)}
    x_base_backward = set() if xend is None else {round(xend, digits)}
    x_base_twoside = add_x_set | {round(xmin, digits), round(xmax, digits)}

    def expand(bases, steps):
        seen = set(bases)
        frontier = set(bases)

        for _ in range(T):
            next_frontier = set()

            for x in frontier:
                for step in steps:
                    y = round(x + step, digits)

                    if xmin - eps <= y <= xmax + eps and y not in seen:
                        next_frontier.add(y)

            if not next_frontier:
                break

            seen.update(next_frontier)
            frontier = next_frontier

        return seen

    s_forward = expand(x_base_forward, steps_forward)
    s_backward = expand(x_base_backward, steps_backward)

    # Preserve the original meanings of base+net and base-net without mixing both step types along one path
    s_twoside = (
        expand(x_base_twoside, steps_forward)
        | expand(x_base_twoside, steps_backward)
    )

    s_e = s_forward | s_backward | s_twoside

    print(len(s_e), "个候选点构建完成。")
    return np.array(sorted(s_e), dtype=float)

def filter_layered_graph(T, states, transitions, req_init, req_by_t=None, eps=1e-10):
    """Filter layered graph nodes and transitions by state requirements."""
    print("开始对有向图根据初始状态以及指定时段要求进行筛选...")
    st = time.time()
    nodes_by_t = {}

    nodes_init = []
    if req_init:
        nodes_init = filter_states_by_req(states, req_init, eps)
    else:
        # nodes_init = [s.id for s in states]  # Without an initial-state requirement, every state may be initial
        nodes_init = None  # Omit the initial layer when no initial-state requirement is specified

    if nodes_init:
        nodes_by_t[-1] = set(nodes_init)

    all_state_ids = {s.id for s in states}

    # for t in range(T):
    for t in range(T):
        req = req_by_t.get(t) if req_by_t is not None else None
        if req is None:
            nodes_by_t[t] = set(all_state_ids)
        else:
            nodes_by_t[t] = set(filter_states_by_req(states, req, eps))

    nodes_virtual_end = set(all_state_ids)  # The virtual terminal layer contains all states
    nodes_by_t[T] = nodes_virtual_end  # Add the virtual terminal layer

    t0 = min(nodes_by_t.keys())  # Index of the initial network layer
    t_end = T+1  # Number of periods, including the virtual terminal layer

    transitions_from = {}
    transitions_to = {}
    transition_order = {}

    for k, (i, j) in enumerate(transitions):
        transitions_from.setdefault(i, set()).add(j)
        transitions_to.setdefault(j, set()).add(i)
        transition_order[(i, j)] = k

    print(f"到开始筛选用时{time.time() - st:.2f}秒。")

    for t in range(t0 + 1, t_end - 1):
        reachable =  set()
        for i in nodes_by_t[t - 1]:
            reachable.update(transitions_from.get(i, ()))
        nodes_by_t[t] &= reachable
        if not nodes_by_t[t]:
            raise ValueError(f"No available state at layer {t}.")

    for t in range(t_end - 2, t0 - 1, -1):
        can_reach = set()
        for j in nodes_by_t[t + 1]:
            can_reach.update(transitions_to.get(j, ()))
        nodes_by_t[t] &= can_reach
        if not nodes_by_t[t]:
            raise ValueError(f"No available state at layer {t}.")
    print(f"向前和向后筛选完成，耗时{time.time() - st:.2f}秒。")

    edges_by_t = []

    for t in range(t0, t_end - 1):
        to_nodes = nodes_by_t[t + 1]
        for i in nodes_by_t[t]:
            edges_by_t.extend((i, j, t) for j in transitions_from[i] if j in to_nodes)

    nodes_by_t = {t: sorted(nodes) for t, nodes in nodes_by_t.items()}
    print(f"有向图筛选完成，总耗时 {time.time() - st:.2f} 秒。")

    return nodes_by_t, edges_by_t

def filter_states_by_req(states, req, eps=1e-10):
    """Filter the states in one layer according to req."""
    nodes = []
    number_types = (int, float, np.integer, np.floating)

    for s in states:
        matched = True
        for key, val in req.items():
            if val is None:
                continue

            state_val = getattr(s, key)
            if isinstance(state_val, number_types) and isinstance(val, number_types):
                if abs(state_val - val) > eps:
                    matched = False
                    break
            elif state_val != val:
                matched = False
                break

        if matched:
            nodes.append(s.id)

    return nodes
