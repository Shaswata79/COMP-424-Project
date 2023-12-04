# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from queue import Queue
import random



@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        main_start_time = time.time()
        copy_board = deepcopy(chess_board)
        copy_mypos = deepcopy(my_pos)
        copy_advpos = deepcopy(adv_pos)
        pos, dir = monte_carlo_tree_search(copy_board, copy_mypos, copy_advpos, max_step, 5, 30, main_start_time)
        main_end_time = time.time()
        time_taken = (main_end_time - main_start_time)
        print("My AI's turn took ", time_taken, "seconds.")

        return pos, dir


moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
opposites = {0: 2, 1: 3, 2: 0, 3: 1}    

# num_simulations = 5, max_chosen_moves = 30
def monte_carlo_tree_search(chess_board, my_pos, adv_pos, max_step, num_simulations, max_chosen_moves, start_time):
    sim = 0
    # get possible moves
    valid_moves = get_possible_moves(deepcopy(chess_board), deepcopy(my_pos), deepcopy(adv_pos), max_step)
    
    # if total possible moves more than some maximum N, choose N unique moves         
    chosen_moves = []
    if len(valid_moves) > int(max_chosen_moves):
        chosen_moves = random.sample(valid_moves, int(max_chosen_moves))
    else:
        chosen_moves = valid_moves
        
    end_time = time.time()
    score_list = [(0, 0) for _ in range(len(chosen_moves))]     # (total_sims, total_score)
    mcts_scores = [0 for _ in range(len(chosen_moves))]         # score for each move
    
    # repeat until time limit is reached
    time_remaining = (2 - (end_time - start_time))
    while time_remaining > 0.5:
        # for each of the 30 child_node of parent:
        for i in range (0, len(chosen_moves)):
            copy_chess_board = deepcopy(chess_board)
            my_new_pos = chosen_moves[i][0]
            barrier_dir = chosen_moves[i][1]
            set_barrier(copy_chess_board, my_new_pos[0], my_new_pos[1], barrier_dir)
            score = 0
            # perform 5 random simulations from current child_node using default policy (5 simulations/child_node, total of 150 random simulations)
            for j in range(0, int(num_simulations)):
                # update score of child_node
                score += simulate(copy_chess_board, my_new_pos, deepcopy(adv_pos), max_step, start_time)
                sim += 1
                end_time = time.time()    
                time_remaining = (2 - (end_time - start_time))
                if time_remaining < 0.4:
                    break
            score_list[i] = (score_list[i][0]+num_simulations, score_list[i][1]+score)
            end_time = time.time()    
            time_remaining = (2 - (end_time - start_time))
            # if time_remaining < 0.4 then stop, else continue with another iteration using the same 30 child_nodes 
            if time_remaining < 0.4:
                break
    
    print("Run Statistics")
    print(f"Number of moves: {len(chosen_moves)}, Time remaining: {time_remaining}, Total simulations: {sim}")
    if len(score_list) == 0:
        print(f"Error, number of moves is 0!! Chess Board Length: {chess_board}, my_pos: {my_pos}, adv_pos: {adv_pos}")
        
    # calculate scores
    for i in range(len(mcts_scores)):
        total_sims, total_score = score_list[i]
        mcts_scores[i] = total_score / total_sims
    # the child_node with the highest simulation score is chosen as the next move 
    index = mcts_scores.index(max(mcts_scores))
    return chosen_moves[index]


# class to keep track of game board state, player position and adversary position
class State():    
    def __init__(self, chess_board, my_pos, adv_pos):
        self.my_pos = deepcopy(my_pos)
        self.adv_pos = deepcopy(adv_pos)
        self.chess_board = deepcopy(chess_board)
        
    def print_state(self):
        print(f"Chess Board Length: {self.chess_board}, my_pos: {self.my_pos}, adv_pos: {self.adv_pos}")
    

# HELPER FUNCTIONS

# finds the possible moves for player (takes ~2 ms)
def get_possible_moves(chess_board, my_pos, adv_pos, max_step):
    unique_pos = set()
    possible_pos = []
    bfs_queue = Queue()
    bfs_queue.put((my_pos[0], my_pos[1], 0))
    
    while True:
        if bfs_queue.empty():
            break
        current_pos = bfs_queue.get()
        r, c, step = current_pos
        for d in range (0, 4):                                  # choose direction to move
            if not chess_board[r,c,d] and not adv_pos == (r+moves[d][0], c+moves[d][1]) and step < max_step:
                m_r, m_c = (r+moves[d][0], c+moves[d][1])       # valid direction for 1 step
                if not (my_pos == (m_r, m_c)):                  # do not put original pos in queue
                    bfs_queue.put((m_r, m_c, step+1)) 
                for b in range(0, 4):                           # choose direction to put barrier
                    if (m_r, m_c, b) not in unique_pos and not chess_board[m_r,m_c,b]:
                        possible_pos.append(((m_r, m_c), b))
                        unique_pos.add((m_r, m_c, b))       
    return possible_pos

# performs random simulations until end of game
def simulate(chess_board, my_pos, adv_pos, max_step, start):
    state = State(chess_board, my_pos, adv_pos)
    turn = 0
    score = 0

    while True:
        end = time.time()
        time_remaining = (2 - (end - start))
        if time_remaining < 0.3:
            print("Simulation timeout!")
            return 0
        
        if turn % 2 == 0:   # my turn
            state = player_move(state, max_step)
            if state == -1:
                return 0
        else:               # opponent's turn
            state = opponent_move(state, max_step)
            if state == -1:
                return 0
        turn += 1
        
        # at every turn, check for endgame and return if true
        game_end, my_score, adv_score = check_endgame(state.chess_board, len(state.chess_board), state.my_pos, state.adv_pos)  
        if game_end:
            score = calculate_simulation_score(my_score, adv_score)
            break
    # after end of game (reached leaf node), if win return 1 else return 0
    return score


def player_move(state, max_step):
    move = random_move(state.chess_board, state.my_pos, state.adv_pos, max_step)
    if move is not None:
        new_pos, dir = move
    else:
        return -1
    set_barrier(state.chess_board, new_pos[0], new_pos[1], dir)
    state.my_pos = new_pos
    return state

def opponent_move(state, max_step):
    move = random_move(state.chess_board, state.adv_pos, state.my_pos, max_step)
    if move is not None:
        new_pos, dir = move
    else:
        return -1
    set_barrier(state.chess_board, new_pos[0], new_pos[1], dir)
    state.adv_pos = new_pos
    return state

# Return 1 for win, 0 for loss/draw
def calculate_simulation_score(my_score, adv_score):
    if int(my_score) > int(adv_score):
        return 1
    return 0


# check if end of game (copied from world.py) 
def check_endgame(chess_board, board_size, p0_pos, p1_pos):
    """
    Check if the game ends and compute the current score of the agents.

    Returns
    -------
    is_endgame : bool
        Whether the game ends.
    player_1_score : int
        The score of player 1.
    player_2_score : int
        The score of player 2.
    """
    # Union-Find
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                moves[1:3]
            ):  # Only check down and right
                if chess_board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))
    p0_r = find(tuple(p0_pos))
    p1_r = find(tuple(p1_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, p0_score, p1_score
    return True, p0_score, p1_score



# set barrier for given direction and position (copied from world.py)
def set_barrier(chess_board, r, c, dir):
    # Set the barrier to True
    chess_board[r, c, dir] = True
    # Set the opposite barrier to True
    move = moves[dir]
    chess_board[r + move[0], c + move[1], opposites[dir]] = True


# Random move (copied from random_agent.py)
def random_move(chess_board, my_pos, adv_pos, max_step):
    steps = np.random.randint(0, max_step + 1)

    # Pick steps random but allowable moves
    for _ in range(steps):
        r, c = my_pos
        # Build a list of the moves we can make
        allowed_dirs = [ d                                
            for d in range(0,4)                           # 4 moves possible
            if not chess_board[r,c,d] and                 # chess_board True means wall
            not adv_pos == (r+moves[d][0],c+moves[d][1])] # cannot move through Adversary
        
        if len(allowed_dirs)==0:
            # If no possible move, we must be enclosed by our Adversary
            break
        random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]
        m_r, m_c = moves[random_dir]
        my_pos = (r + m_r, c + m_c)

        # Pick where to put our new barrier, at random
        r, c = my_pos
        allowed_barriers=[i for i in range(0,4) if not chess_board[r,c,i]]
        assert len(allowed_barriers)>=1 
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

        return my_pos, dir

