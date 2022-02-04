# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        next_values = util.Counter()
        for iterate in range(0, self.iterations):
            for state in self.mdp.getStates():
                val = -float("inf")
                if self.mdp.isTerminal(state):
                    val = 0

                for action in self.mdp.getPossibleActions(state):
                    if(val < (self.computeQValueFromValues(state, action))):
                        val = (self.computeQValueFromValues(state, action))

                next_values[state] = val

            self.values = next_values.copy()



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0;
        for t in self.mdp.getTransitionStatesAndProbs(state, action):
            value = value + (t[1] * (self.mdp.getReward(state, action, t[0]) + self.discount * self.values[t[0]]))
        return value
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        q_v = -float("inf")
        next_action = None
        for act in self.mdp.getPossibleActions(state):
            value = self.computeQValueFromValues(state,act)
            if(value > q_v):
                q_v = value
                next_action = act

        return next_action

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for iterate in range(0,self.iterations):
            indexOfState = iterate % len(self.mdp.getStates())
            state = self.mdp.getStates()[indexOfState]
            if self.mdp.isTerminal(state):
                continue

            best_q_value = -float("inf")
            for action in self.mdp.getPossibleActions(state):
                value = self.computeQValueFromValues(state, action)
                if best_q_value < value:
                    best_q_value = value
            self.values[state] = best_q_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        p_queue = util.PriorityQueue()
        predecessors  = {}

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for stt,_ in self.mdp.getTransitionStatesAndProbs(state, action):
                    if stt in predecessors :
                        predecessors [stt].add(state)
                    else:
                        predecessors [stt] = {state}


        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            best_val = -float("inf")
            for action in self.mdp.getPossibleActions(state):
                value = self.computeQValueFromValues(state,action)
                if(best_val < value):
                    best_val = value

            diff = abs(self.values[state] - best_val)
            p_queue.update(state, -diff)

        for iterate in range(0,self.iterations):
            if p_queue.isEmpty():
                break
            state = p_queue.pop()
            if not self.mdp.isTerminal(state):
                best_val = -float("inf")
                for action in self.mdp.getPossibleActions(state):
                    value = self.computeQValueFromValues(state, action)
                    if (best_val < value):
                        best_val = value
                self.values[state] = best_val

            for predecessor  in predecessors [state]:
                if self.mdp.isTerminal(predecessor ):
                    continue
                else:
                    best_val = -float("inf")
                    for action in self.mdp.getPossibleActions(predecessor):
                        value = self.computeQValueFromValues(predecessor, action)
                        if (best_val < value):
                            best_val = value
                    difff = abs(self.values[predecessor ] - best_val)
                    if difff > self.theta:
                            p_queue.update(predecessor , -difff)

