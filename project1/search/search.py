# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    tuples = util.Stack()

    currentState = (problem.getStartState(),[])
    tuples.push(currentState)

    visiteds = list()

    while not (tuples.isEmpty()):
        currentState = tuples.pop()
        currentStateLocation = currentState[0]
        currentStatePath = currentState[1]

        if problem.isGoalState(currentStateLocation):
            return currentStatePath

        if not currentStateLocation in visiteds:
            visiteds.append(currentStateLocation)

            for adj in list(problem.getSuccessors(currentStateLocation)):
                if not adj[0] in visiteds:
                    path = currentStatePath+[adj[1]]
                    nextState = (adj[0], path)
                    tuples.push(nextState)
    util.raiseNotDefined()




def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    currentState = (problem.getStartState(),[])
    visiteds = list()

    queueNodes = util.Queue()
    queueNodes.push(currentState)


    if (problem.isGoalState(problem.getStartState())):
        return [];

    while not queueNodes.isEmpty():
        currentState = queueNodes.pop()
        currentStateLocation = currentState[0]
        currentStatePath = currentState[1]


        if (problem.isGoalState(currentStateLocation)):
            return currentStatePath;

        if not currentStateLocation in visiteds:
            visiteds.append(currentStateLocation)
            for adj in problem.getSuccessors(currentStateLocation):
                if not adj[0] in visiteds:
                    newState = (adj[0],currentStatePath+[adj[1]])
                    queueNodes.push(newState)
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    priorityQueue = util.PriorityQueue()
    currentState = (problem.getStartState(), [], 0)  # the state,path and cost
    priorityQueue.push(currentState, 0)

    visited = []

    if (problem.isGoalState(problem.getStartState())):
        return[];

    while not priorityQueue.isEmpty():
        currentState = priorityQueue.pop()
        currentStateCost = currentState[2]
        currentStatePath = currentState[1]
        currentStateLocation = currentState[0]

        if (problem.isGoalState(currentStateLocation)):
            return currentStatePath;

        if not currentStateLocation in visited:
            visited.append(currentStateLocation)
            for adj in problem.getSuccessors(currentStateLocation):
                if not adj[0] in visited:
                    nextNode = (adj[0], currentStatePath + [adj[1]], currentStateCost + adj[2])
                    nextNodeCost = currentStateCost + adj[2]
                    priorityQueue.push(nextNode, nextNodeCost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()
    currentState = (problem.getStartState(), [], 0)

    priorityQueue.push(currentState, heuristic(currentState[0],problem)+currentState[2])

    visiteds = []

    if problem.isGoalState(currentState[0]):
        return[];

    while not priorityQueue.isEmpty():
        currentState = priorityQueue.pop()
        currentStateLocation = currentState[0]
        currentStatePath = currentState[1]
        currentStateCost = currentState[2]

        if problem.isGoalState(currentStateLocation):
            return currentStatePath

        if not currentStateLocation in visiteds:
            visiteds.append(currentStateLocation)
            for adj in problem.getSuccessors(currentStateLocation):
                if not adj[0] in visiteds:
                    nextNode = (adj[0], currentStatePath + [adj[1]], currentStateCost + adj[2])
                    priorityQueue.push(nextNode, currentStateCost + adj[2] + heuristic(adj[0], problem))

    util.raiseNotDefined()



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
