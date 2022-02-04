# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #we want to maximize the ghost distances and minimize food distances
        #but if we increase ghost distances too much , probably pacman will loop and ....

        FoodDistances = []
        GhostDistances = []

        Food = newFood.asList()
        ghostPositions = successorGameState.getGhostPositions()

        if len(Food) == 0:
            return +float("inf")

        if currentGameState.getPacmanPosition() == newPos:
            return -float("inf")

        for food in Food:
            FoodDistances.append(manhattanDistance(food, newPos))
        for ghost in ghostPositions:
            GhostDistances.append(manhattanDistance(ghost, newPos))

        for distance in GhostDistances:
            if distance == 1 or distance == 0:
                return -float("inf")

        sumDistances = sum(FoodDistances)
        avgDistance = sumDistances/len(food)


        return  (800/ avgDistance) + (5000 / len(FoodDistances))


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_agent(gameState,depth):
            if( gameState.isWin() or gameState.isLose() or depth == self.depth):
                return (self.evaluationFunction(gameState),None);
            maximum = -float("inf")
            pacmanActions = gameState.getLegalActions(0)
            if(len(pacmanActions) == 0):
                return (self.evaluationFunction(gameState),None);

            bestAct = pacmanActions[0]

            for action in pacmanActions:
                value = min_agent(gameState.generateSuccessor(0, action),1,depth)[0]
                if (maximum < value):
                    maximum = value
                    bestAct = action

            return (maximum,bestAct);

        def min_agent(gameState,ghostID,depth):
            numberOfAgents = gameState.getNumAgents()
            minimum = +float("inf")
            ghostActions = gameState.getLegalActions(ghostID)

            if(depth == self.depth):
                return (self.evaluationFunction(gameState),None);

            if(len(ghostActions) == 0):
                return (self.evaluationFunction(gameState),None);

            bestAct = ghostActions[0]

            for action in ghostActions:
                if(ghostID == numberOfAgents-1):
                    value = max_agent(gameState.generateSuccessor(ghostID, action),depth+1)[0]
                    if (minimum>value):
                        minimum = value
                        bestAct = action
                else:
                    value = min_agent(gameState.generateSuccessor(ghostID, action),ghostID+1,depth)[0]
                    if (minimum>value):
                        minimum = value
                        bestAct = action

            return (minimum,bestAct);

        return max_agent(gameState,0)[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"


        def max_agent(gameState,depth,alpha,beta):

            if( gameState.isWin() or gameState.isLose() or depth == self.depth):
                return (self.evaluationFunction(gameState),None);
            maximum = -float("inf")
            pacmanActions = gameState.getLegalActions(0)
            if(len(pacmanActions) == 0):
                return (self.evaluationFunction(gameState),None);

            bestAct = pacmanActions[0]

            for action in pacmanActions:
                value = min_agent(gameState.generateSuccessor(0, action),1,depth,alpha,beta)[0]
                if (maximum < value):
                    maximum = value
                    bestAct = action
                    if (alpha <maximum):
                        alpha = maximum
                if (alpha > beta):
                    return (maximum,bestAct)

            return (maximum,bestAct);

        def min_agent(gameState,ghostID,depth,alpha,beta):
            numberOfAgents = gameState.getNumAgents()
            minimum = +float("inf")
            ghostActions = gameState.getLegalActions(ghostID)

            if(depth == self.depth):
                return (self.evaluationFunction(gameState),None);

            if(len(ghostActions) == 0):
                return (self.evaluationFunction(gameState),None);

            bestAct = ghostActions[0]

            for action in ghostActions:
                if(ghostID == numberOfAgents-1):
                    value = max_agent(gameState.generateSuccessor(ghostID, action),depth+1,alpha,beta)[0]
                    if (minimum>value):
                        minimum = value
                        bestAct = action
                        if (beta > minimum):
                            beta = minimum
                    if(alpha > beta):
                        return (minimum,bestAct);
                else:
                    value = min_agent(gameState.generateSuccessor(ghostID, action),ghostID+1,depth,alpha ,beta)[0]
                    if (minimum>value):
                        minimum = value
                        bestAct = action
                        if(beta > minimum):
                            beta = minimum
                    if (alpha > beta):
                        return (minimum, bestAct);

            return (minimum,bestAct);

        return max_agent(gameState,0,-float("inf"),+float("inf"))[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_agent(gameState, depth):
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                return (self.evaluationFunction(gameState), None);
            maximum = -float("inf")
            pacmanActions = gameState.getLegalActions(0)
            if (len(pacmanActions) == 0):
                return (self.evaluationFunction(gameState), None);

            bestAct = pacmanActions[0]

            for action in pacmanActions:
                value = min_agent(gameState.generateSuccessor(0, action), 1, depth)[0]
                if (maximum < value):
                    maximum = value
                    bestAct = action

            return (maximum, bestAct);

        def min_agent(gameState, ghostID, depth):
            numberOfAgents = gameState.getNumAgents()
            avg = 0
            ghostActions = gameState.getLegalActions(ghostID)

            if (depth == self.depth):
                return (self.evaluationFunction(gameState), None);

            if (len(ghostActions) == 0):
                return (self.evaluationFunction(gameState), None);

            avgAction = ghostActions[0]
            probability = 1 / len(ghostActions)

            for action in ghostActions:
                if (ghostID == numberOfAgents - 1):
                    value = max_agent(gameState.generateSuccessor(ghostID, action), depth + 1)[0]
                    avg = avg + probability * value
                else:
                    value = min_agent(gameState.generateSuccessor(ghostID, action), ghostID + 1, depth)[0]
                    avg = avg + probability * value

            #the action does not matter . the avg value is important . so we dont know the avgAction
            # but we have avg
            return (avg, avgAction);

        return max_agent(gameState, 0)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    first of all I want to divide ghosts to two kind of ghosts:
    first , the harmless ghosts and second , the harmful ghosts
    the hramless ghost are two kind:
    1- pacman can eat them (scared time > distance) which is very good
    2- pacman can not eat them but it can ignore them for awhile

    after that we want to :
    decrease the number of foods available so
    increase the distance to ghosts
    decrease the distance to the foods
    decrease the capsules

    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin():
        return +float("inf")
    elif currentGameState.isLose():
        return -float("inf")


    pacmanPos = currentGameState.getPacmanPosition()
    foodsList = currentGameState.getFood().asList();
    capsules = currentGameState.getCapsules();
    ghostsList = currentGameState.getGhostStates()

    minDistanceGhost = +float("inf")
    closestGhost = ghostsList[0]

    score = 0

    for ghost in ghostsList:
        value = manhattanDistance(pacmanPos,ghost.configuration.pos)
        if(value <= 1):
            if(ghost.scaredTimer >=1):
                score += 100
            else:
                score -=300
        minDistanceGhost = min(minDistanceGhost,value)
        closestGhost = ghost

    if(ghost.scaredTimer < minDistanceGhost):
        score -= (4/minDistanceGhost)

    minimumDistanceFood = +float("inf")

    if len(foodsList) == 0:
        return +float("inf")

    closesFood = foodsList[0]
    for food in foodsList:
        value = manhattanDistance(pacmanPos,food)
        minimumDistanceFood = min(value,minimumDistanceFood)
        closesFood = food

    score -= (2*minimumDistanceFood)
    #less number of foods available is better...
    score -= (10*len(foodsList))

    minimumDistanceCapsule = +float("inf")
    if(len(capsules)>=1):
        closestCapsule = capsules[0]
        for cap in capsules:
            value = manhattanDistance(cap,pacmanPos)
            minimumDistanceCapsule = min(value,minimumDistanceCapsule)
            closestCapsule = cap
        score -= minimumDistanceCapsule

    score -= 5*len(capsules)

    score += scoreEvaluationFunction(currentGameState)
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
