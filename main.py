from utils import get_stage, make_profile
from mcts_by_step import MCTSByStep
from mcts import MCTS
import tracemalloc

s = get_stage(number=1)
# solver = MCTSByStep(s)
# solution, is_goal = solver.solve(simulation_time=1)
# if is_goal:
#     for gameState in solution:
#         print(gameState)
path_list = []
memory_list = []
time_list = []
for i in range(10):
    tracemalloc.start()
    solver = MCTS(s, simulation_time=20, max_simulation_depth=30, max_previous_nodes=2, points_list=None,
                  is_unique_node=False, c=1.4)

    solution, time, explored_nodes_num = make_profile(solver)
    if solution[-1].is_goal():
        print("Won !!")
    path_list.append(len(solution))
    memory_list.append(tracemalloc.get_traced_memory()[1])
    time_list.append(time)

    print(solution[-1])
    # print(len(solution))
    # print(time)
    # print(explored_nodes_num)
    # print(tracemalloc.get_traced_memory())
    tracemalloc.stop()

print(path_list)
print(memory_list)
print(time_list)
