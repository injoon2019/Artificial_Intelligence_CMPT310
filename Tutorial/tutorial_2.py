# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:41:09 2020

@author: mahkh
"""
import sys
sys.path.insert(0, 'C:\\Users\\injoo\\Desktop\\SFU\\2020.5 Sixth Semester\\CMPT 310\\aima-python-master\\aima-python-master')
# In Linux: sys.path.insert(0, '/local-scratch/aima-python')

from search import * # This should run without error messages

#def make_rand_8puzzle():
#    # generate a random initial state here:
#    # ...
#    return EightPuzzle(initial)

initial=(1, 2, 3, 0, 5, 6, 4, 7, 8)
goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)
puz = EightPuzzle(initial, goal)

state=(1, 2, 3, 0, 5, 6, 4, 7, 8)
print(puz.check_solvability(state))
state=(1, 0, 3, 4, 5, 6, 7, 2, 8)
print(puz.check_solvability(state))



initial=(1, 2, 3, 0, 5, 6, 4, 7, 8)
goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)
problem_instance = EightPuzzle(initial, goal)
node = astar_search(problem_instance, h=problem_instance.h)
print('Solution:', node.solution())
print('Path:', node.path())
print(len(node.path()))


initial=(7, 2, 4, 5, 0, 6, 8, 3, 1)
goal =(0, 1, 2, 3, 4, 5, 6, 7, 8)

# g(n): the cost to reach the node n (the path cost from the start node to node n)
# h(n): the estimated cost to get from node n to the goal
# A∗ search is a form of best-first search. It tries first the node with the
# lowest value of f(n) = g(n) + h(n).

# Conditions for optimality: Admissibility and consistency
# number of misplaced tiles: h1 = 8
# Manhattan distance: h2 = 3 + 1 + 2 + 2 + 2 + 3 + 3 + 2 = 18
# An admissible heuristic is one that never overestimates the cost to reach the goal.

# As expected, neither of these overestimates the true solution cost, which is 26.



class VacuumCleaner(Problem):
    """ The vacuum-cleaner world is a 4x4 grid of locations.
    Each location in the grid is either clean or dirty. A vacuum-cleaner is in
    one of the locations. A state is represented as a tuple of length 17, where
    element at index i (0<= i <=15) is either 0 (clean) or 1 (dirty), and the last
    element at index 16 represents the location of the vacuum-cleaner, which is
    a number between 0 to 16. There are 16 * 2^16 possible states.
    """

    def __init__(self, initial, goal=tuple([0] * 17)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_vacuum_cleaner(self, state):
        """Return the cocation of the vacuum-cleaner in a given state"""

        return state[16]

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only five possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SUCK']
        loc = self.find_vacuum_cleaner(state)

        if state[loc] == 0:
            possible_actions.remove('SUCK')
            # pass

        if loc % 4 == 0:
            possible_actions.remove('LEFT')
        if loc < 4:
            possible_actions.remove('UP')
        if loc % 4 == 3:
            possible_actions.remove('RIGHT')
        if loc > 11:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # loc is the index of the vacuum cleaner square
        loc = self.find_vacuum_cleaner(state)
        new_state = list(state)
        if action == 'SUCK':
            new_state[loc] = 0
        else:
            delta = {'UP': -4, 'DOWN': 4, 'LEFT': -1, 'RIGHT': 1}
            new_state[16] = loc + delta[action]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        return True

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        return sum(s != g for (s, g) in zip(node.state, self.goal))

        #return 0






#0    0    0    0
#0    0    0    0
#0    0    0    0
#0    0    0x   0
initial=[0] * 17
initial[16] = 14
initial = tuple(initial)


def display(state):
    for i in range(4):
        for j in range(4):
            if 4*i+j == state[16]:
                print(str(state[4*i+j]) + 'x', end="   ")
            else:
                print(state[4*i+j], end="    ")
        print()

display(initial)

vc_problem = VacuumCleaner(initial)


node = astar_search(vc_problem, h=vc_problem.h)
print('Solution:', node.solution())
print('Path:', node.path())

for n in node.path():
    display(n.state)
    print('_________________')


#0    0    0    0
#0    0x   0    0
#0    0    0    0
#0    1    1    0
initial=[0] * 17
initial[13] = 1
initial[14] = 1
initial[16] = 5
initial = tuple(initial)
vc_problem = VacuumCleaner(initial)
node = astar_search(vc_problem, h=vc_problem.h)
print('Solution:', node.solution())

for n in node.path():
    display(n.state)
    print('_________________')


#1x   0    0    1
#0    0    0    0
#1    1    1    1
#0    1    1    0
initial=[0] * 17
initial[0] = 1
initial[3] = 1
initial[8] = 1
initial[9] = 1
initial[10] = 1
initial[11] = 1
initial[13] = 1
initial[14] = 1
initial[16] = 0
initial = tuple(initial)
vc_problem = VacuumCleaner(initial)
node = astar_search(vc_problem, h=vc_problem.h)
print('Solution:', node.solution())
for n in node.path():
    display(n.state)
    print('_________________')




# Graph class from Data Structure and Algorithmic Thinking with Python
# by Narasimha Karumanchi:
class Vertex:
    def __init__(self, id):
        self.id = id
        self.adjacent = {}  # key: instance of the Vertex, value: weight
        self.visit = False
        self.inDegree = 0
        # self.previous = None
        # self.distance = sys.maxint
    def addNeighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight   # neighbor is an instance of the Vertex
class Graph:
    def __init__(self):
        self.vertexDict = {}  # key: id, value: instance of the Vertex
        self.numVertices = 0
    def addVertex(self, node):
        self.numVertices += 1
        newVertex = Vertex(node)
        self.vertexDict[node] = newVertex
        return newVertex
    def getVertices(self):
        return self.vertexDict.keys()
    def getEdges(self):
        edges = []
        for u in self.vertexDict.values():
            for v in u.adjacent:
                edges.append((u.id, v.id, u.adjacent[v]))
        return edges
    def addEdge(self, frm, to, cost=0):
        if frm not in self.vertexDict:
            self.addVertex(frm)
        if to not in self.vertexDict:
            self.addVertex(to)
        self.vertexDict[frm].adjacent[self.vertexDict[to]] = cost
        self.vertexDict[to].inDegree += 1
        # for directed graphs do not add this:
        # self.vertexDict[to].adjacent[self.vertexDict[frm]] = cost


G = Graph()
for i in range(10):
    G.addVertex(i)

G.addEdge(0,5);G.addEdge(0,1);G.addEdge(1,6);G.addEdge(1,2);G.addEdge(2,7);
G.addEdge(2,3);G.addEdge(3,8);G.addEdge(3,4)
G.addEdge(4,9);G.addEdge(4,0);G.addEdge(5,8);G.addEdge(6,9);G.addEdge(7,5);
G.addEdge(8,6);G.addEdge(9,7);
print(G.getEdges())
def dfs(G, u):
    if not u.visit:
        print(u.id,"visited")
        u.visit = True
        for v in u.adjacent:
            dfs(G, v)
def dfsTraversal(G):
    for u in G.vertexDict.values():
        dfs(G, u)
from collections import deque
def bfs(G, u):
    if not u.visit:
        q = deque([u])
        while len(q) > 0:
            u = q.popleft()
            print(u.id,"visited")
            u.visit = True
            for v in u.adjacent:
                if not v.visit:
                    q.append(v)

def bfsTraversal(G):
    for u in G.vertexDict.values():
        bfs(G, u)
bfsTraversal(G)
#dfsTraversal(G)


def topological_sort(G):
    order = []
    q = deque([])
    for u in G.vertexDict.values():
        if u.inDegree == 0:
           q.append(u)
    while len(q) > 0:
        u = q.popleft()
        order.append(u.id)
        for v in u.adjacent:
            v.inDegree -= 1
            if v.inDegree == 0:
                q.append(v)
    return order

G = Graph()
for i in range(10):
    G.addVertex(i)
G.addEdge(0,1);G.addEdge(6,1);G.addEdge(1,4);G.addEdge(4,3);G.addEdge(3,2);
G.addEdge(4,2);G.addEdge(2,5);G.addEdge(6,5)
G.addEdge(9,8)
for u in G.vertexDict.values():
    print(u.id,u.inDegree)
print(topological_sort(G))
