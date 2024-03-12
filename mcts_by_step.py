import numpy as np
import random
import time
from collections import deque
from state import State
from state import Action

ROW = 0
COL = 1
C_ABYSS = ' '
C_GREYTILE = '█'
C_HOLE = '#'
C_ORANGETILE = '▒'
C_SOFTSWITCH = 'O'
C_HARDSWITCH = 'X'
C_TELEPORTSWITCH = 'C'


class MCTSNode:
    def __init__(self, game_state: State, parent=None):
        self.gameState = game_state
        self.children = []
        self.parent = parent
        self.untriedStates = None
        self.visitNum = 0
        self.points = -1.0

        def _get_result(current_state: State, mask_list: list):
            if current_state.is_goal():
                return 1.0

            points_list = np.array([0.0, 0.5, -0.5])
            if mask_list[3]:
                return points_list[2]

            if any(x for x in mask_list[0:2]):
                return np.max(points_list[0:2][mask_list[0:2]])

            return 0.0

        self.get_result = _get_result

    @property
    def n(self):
        return self.visitNum

    @property
    def q(self):
        return self.points

    @staticmethod
    def get_rollout_policy(game_state: State, simulation_hash_table: deque):
        actions = list(Action)
        random.shuffle(actions)
        for action in actions:
            test_game_state = game_state.copy()
            if test_game_state.perform(action):
                if not test_game_state.is_end():
                    if test_game_state not in simulation_hash_table:
                        return test_game_state
        returned_gamestate = game_state.copy()
        returned_gamestate.perform(actions[0])
        return returned_gamestate

    def get_untried_actions(self):
        if self.untriedStates is None:
            self.untriedStates = []
            for action in Action:
                test_game_state = self.gameState.copy()
                if test_game_state.perform(action):
                    if not test_game_state.is_end():
                        self.untriedStates.append(test_game_state)
        return self.untriedStates

    def expand(self, hash_table):
        next_game_state = self.get_untried_actions().pop()
        if next_game_state.encode() in hash_table:
            return None
        mcst_child_node = MCTSNode(next_game_state, self)
        self.children.append(mcst_child_node)
        return mcst_child_node

    def copy(self):
        game_state = self.gameState.copy()
        return MCTSNode(game_state)

    def is_terminal_node(self):
        return self.gameState.is_end() or self.gameState.is_goal()

    def get_best_child(self, c=1.4):
        if len(self.children) == 0:
            return None

        if any(child.q != 0.0 for child in self.children):
            c = 0

        weights = [child.q + c * np.sqrt((2 * np.log(self.n)) / child.n) for child in self.children]

        return self.children[np.argmax(weights)]

    def is_fully_expanded(self):
        return len(self.get_untried_actions()) == 0

    @staticmethod
    def handle_result(first_game_state: State, second_game_state: State):
        if not first_game_state.box.is_splitted() and second_game_state.box.is_splitted():
            return 1

        if first_game_state.board.to_string().count(C_GREYTILE) \
                < second_game_state.board.to_string().count(C_GREYTILE):
            return 2
        elif first_game_state.board.to_string().count(C_GREYTILE) \
                > second_game_state.board.to_string().count(C_GREYTILE):
            return 3

        return 0

    def simulate(self, previous_game_state: State, max_simulation_depth=30, max_previous_nodes=4):
        current_game_state = self.gameState

        mask_list = [False, False, False, False]

        index = self.handle_result(previous_game_state, current_game_state)
        if index:
            mask_list[index - 1] = True
        if index == 3:
            mask_list[3] = True

        simulation_hash_table = deque()

        # Tree depth
        depth = 0

        while not current_game_state.is_goal() and not current_game_state.is_end():
            if depth > max_simulation_depth:
                break

            test_game_state = self.get_rollout_policy(current_game_state, simulation_hash_table)

            simulation_hash_table.append(test_game_state)
            if len(simulation_hash_table) > max_previous_nodes:
                simulation_hash_table.popleft()

            index = self.handle_result(current_game_state, test_game_state)

            if index:
                mask_list[index - 1] = True
            if index == 3 and not mask_list[0] and not mask_list[1]:
                mask_list[3] = True
            current_game_state = test_game_state
            depth += 1

        return self.get_result(current_game_state, mask_list)

    def backpropagate(self, result):
        self.visitNum += 1
        self.points = max(self.points, result)
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    def __init__(self, root):
        self.root = root
        self.hashTable = set()
        self.hashTable.add(self.root.gameState.encode())

    def get_best_action(self, simulation_number=None, simulation_time=None, max_simulation_depth=20,
                        max_previous_nodes=2):
        if simulation_number is not None:
            for i in range(simulation_number):
                child = self.tree_policy()
                # if child.gameState.encode() not in self.hashTable:
                #     self.hashTable.add(child.gameState.encode())
                result = child.simulate(child.parent.gameState, max_depth=max_simulation_depth,
                                        max_previous_nodes=max_previous_nodes)
                child.backpropagate(result)
            return self.root.get_best_child().copy()

        elif simulation_time is not None:
            end_time = time.time() + simulation_time
            while True:
                child = self.tree_policy()
                if child.gameState.encode() not in self.hashTable:
                    self.hashTable.add(child.gameState.encode())
                result = child.simulate(child.parent.gameState)
                child.backpropagate(result)
                if time.time() > end_time:
                    break
            return self.root.get_best_child().copy()
        else:
            raise ValueError("No simulation param is provided")

    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                next_node = current_node.expand(self.hashTable)
                if next_node is not None:
                    return next_node
            else:
                next_node = current_node.get_best_child()
                if next_node is not None:
                    current_node = next_node
                else:
                    break
        return current_node

    def __del__(self):
        del self.hashTable


class MCTSByStep:
    def __init__(self, state: State):
        self.gameState = state

    def solve(self, simulation_time=None, simulation_number=None, max_simulation_depth=5,
              max_previous_node=2, max_loop=5000):
        if simulation_number is None and simulation_time is None:
            raise ValueError("Must provide simulation number or simulation time")
        current_node = MCTSNode(self.gameState)
        solution = []
        loop_count = 0
        while True:
            if current_node.gameState.is_goal() or current_node.gameState.is_end():
                break
            if loop_count > max_loop:
                break
            searcher = MCTS(current_node)
            current_node = searcher.get_best_action(simulation_time=simulation_time,
                                                    simulation_number=simulation_number,
                                                    max_simulation_depth=max_simulation_depth,
                                                    max_previous_nodes=max_previous_node)
            loop_count += 1
            solution.append(current_node.gameState)
            print(current_node.gameState)
        if current_node.gameState.is_goal():
            return solution, True
        return solution, False
