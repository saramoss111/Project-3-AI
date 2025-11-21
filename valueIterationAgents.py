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
        for _ in range(self.iterations):
            newValues = util.Counter()

            for state in self.mdp.getStates():

                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue

                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    newValues[state] = 0
                    continue
                qValues = []
                for action in actions:
                    q = self.computeQValueFromValues(state, action)
                    qValues.append(q)

                newValues[state] = max(qValues)

            self.values = newValues

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
        ### Use count for each direction
        result = 0
        sprime_and_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        sprime = []
        prob = []
        reward = {}
        for i in range(len(sprime_and_prob)):
            sprime.append(sprime_and_prob[i][0])
            prob.append(sprime_and_prob[i][1])
            reward[action] = self.mdp.getReward(state, action, sprime[i])
        for j in range(len(sprime_and_prob)):
            if len(sprime_and_prob) == 0:
                continue
            result += prob[j]* (reward[action] + self.discount * self.getValue(sprime[j]))
        return result
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
        if self.mdp.isTerminal(state):
            return None

        possible_actions = self.mdp.getPossibleActions(state)
        value_to_action = {}
        list_of_qvalues = []
        for action in possible_actions:
            q_value = self.computeQValueFromValues(state, action)
            value_to_action[q_value] = action
            list_of_qvalues.append(q_value)
        maximum= max(list_of_qvalues)
        return value_to_action[maximum]

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
        states = self.mdp.getStates()
        num_of_states = len(states)
        for k in range(self.iterations):
            current_value = self.values.copy()
            temp_state = states[k % len(states)]
            if self.mdp.isTerminal(temp_state):
                    continue
            possible_actions = self.mdp.getPossibleActions(temp_state)
            possible_value = []
            for action in possible_actions: #Every Action
                possible_value.append(self.computeQValueFromValues(temp_state, action))
            current_value[temp_state] = max(possible_value)
            self.values = current_value

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
        states = self.mdp.getStates()
        set_of_predecessors = {} #Dictionary of set of predecessors for each state
        pqueue = util.PriorityQueue()
        for state in states: #Initializing the set
            if self.mdp.isTerminal(state):
                continue
            possible_action = self.mdp.getPossibleActions(state)
            reachable_from_state = set()
            for action in possible_action:
                temp = self.mdp.getTransitionStatesAndProbs(state, action)
                for a in range(len(temp)):
                    if temp[a][1] != 0:
                        reachable_from_state.add(temp[a][0])
            for sprime in reachable_from_state:
                if sprime in set_of_predecessors:
                    set_of_predecessors[sprime].add(state)
                else:
                    set_of_predecessors[sprime] = {state}

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            possible_actions = self.mdp.getPossibleActions(state)
            possible_value = []
            for action in possible_actions:
                possible_value.append(self.computeQValueFromValues(state, action))
            q_value = max(possible_value)
            diff = abs(self.values[state] - q_value)
            pqueue.update(state, -diff)

        for i in range(self.iterations):
            if pqueue.isEmpty():
                break
            state = pqueue.pop()
            if self.mdp.isTerminal(state):
                continue
            possible_actions = self.mdp.getPossibleActions(state)
            possible_value2 = []
            for action in possible_actions:
                possible_value2.append(self.computeQValueFromValues(state, action))
            self.values[state] = max(possible_value2)

            for predecessor in set_of_predecessors[state]:
                if self.mdp.isTerminal(predecessor):
                    continue
                possible_actions = self.mdp.getPossibleActions(predecessor)
                possible_value3 = []
                for action in possible_actions:
                    possible_value3.append(self.computeQValueFromValues(predecessor, action))
                max_qvalue = max(possible_value3)
                diff = abs(self.values[predecessor] - max_qvalue)
                if diff > self.theta:
                    pqueue.update(predecessor, -diff)

