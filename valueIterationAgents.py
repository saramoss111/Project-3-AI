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
        result = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        next_states = [s for (s, p) in transitions]
        probs = [p for (s, p) in transitions]
        reward = {action: None}

        for i, s_prime in enumerate(next_states):
            reward[action] = self.mdp.getReward(state, action, s_prime)

        for j, s_prime in enumerate(next_states):
            if len(transitions) == 0:
                continue
            r = reward[action]
            result += probs[j] * (r + self.discount * self.getValue(s_prime))

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

        actions = self.mdp.getPossibleActions(state)
        value_to_action = {}
        q_values = []

        for action in actions:
            q = self.computeQValueFromValues(state, action)
            value_to_action[q] = action
            q_values.append(q)

        best_q = max(q_values)
        return value_to_action[best_q]

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
        num_states = len(states)

        for k in range(self.iterations):
            new_values = self.values.copy()
            state = states[k % num_states]

            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)
            q_values = []

            for action in actions:
                q_values.append(self.computeQValueFromValues(state, action))

            new_values[state] = max(q_values)
            self.values = new_values

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
        predecessors = {}     
        pq = util.PriorityQueue()

        for state in states:
            if self.mdp.isTerminal(state):
                continue

            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob == 0:
                        continue
                    if next_state not in predecessors:
                        predecessors[next_state] = set()
                    predecessors[next_state].add(state)

        for state in states:
            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)
            if not actions:
                continue

            q_values = [self.computeQValueFromValues(state, action) for action in actions]
            max_q = max(q_values)
            diff = abs(self.values[state] - max_q)
            pq.update(state, -diff)   

        for _ in range(self.iterations):
            if pq.isEmpty():
                break

            state = pq.pop()
            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)
            if actions:
                q_values = [self.computeQValueFromValues(state, action) for action in actions]
                self.values[state] = max(q_values)

            if state not in predecessors:
                continue

            for pred in predecessors[state]:
                if self.mdp.isTerminal(pred):
                    continue

                actions = self.mdp.getPossibleActions(pred)
                if not actions:
                    continue

                q_values = [self.computeQValueFromValues(pred, action) for action in actions]
                max_q = max(q_values)
                diff = abs(self.values[pred] - max_q)

                if diff > self.theta:
                    pq.update(pred, -diff)