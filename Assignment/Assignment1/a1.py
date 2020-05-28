"""
    Class: CMPT 310
    Name: Injun Son
    Date: May 29, 2020
    Assignment1 : Experementing with the 8-puzzle
"""

#a1.py
import sys
sys.path.insert(0, 'C:\\Users\\injoo\\Desktop\\SFU\\2020.5 Sixth Semester\\CMPT 310\\aima-python-master\\aima-python-master')
from search import *
import random

def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

    print("best_first_graph_search called")

    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None



#-----------------------

#Question1: Helper functions
def make_rand_8puzzle():
    state = list(range(9))
    random.shuffle(state)
    eightPuzzle = EightPuzzle(tuple(state))

    while (eightPuzzle.check_solvability(state)== False):
        random.shuffle(state)
        eightPuzzle = EightPuzzle(tuple(state))

    return eightPuzzle


def display(state):
    for i in range(len(state)):
        if(state[i]==0):
            print('*', end=" ")
        else:
            print(state[i], end=" ")
        if(i%3== 2):
            print()


#---------------------------------------------
# g(n): the cost to reach the node n (the path cost from the start node to node n)
# h(n): the estimated cost to get from node n to the goal
# A∗ search is a form of best-first search. It tries first the node with the
# lowest value of f(n) = g(n) + h(n).

# Conditions for optimality: Admissibility and consistency
# number of misplaced tiles: h1 = 8
# Manhattan distance: h2 = 3 + 1 + 2 + 2 + 2 + 3 + 3 + 2 = 18
# An admissible heuristic is one that never overestimates the cost to reach the goal.

# As expected, neither of these overestimates the true solution cost, which is 26.
#Question2
import time
for i in range(0,1):
    eightPuzzle = make_rand_8puzzle()
    manhAstar = eightPuzzle
    maxAstar = eightPuzzle

    display(eightPuzzle.initial)
    startTime = time.time()
    node = astar_search(eightPuzzle, h=eightPuzzle.h)
    elapsedTime = time.time() - startTime
    print('Solution: ', node.solution())
    print("Path: ", node.path())
    print(len(node.path()))
