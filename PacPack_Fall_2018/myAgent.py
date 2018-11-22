# myAgentP3.py
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
# This file was based on the starter code for student bots, and refined
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#########
# Agent #
#########
class MyAgent(CaptureAgent):
  """
  YOUR DESCRIPTION HERE
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    # Make sure you do not delete the following line.
    # If you would like to use Manhattan distances instead
    # of maze distances in order to save on initialization
    # time, please take a look at:
    # CaptureAgent.registerInitialState in captureAgents.py.
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.weights = [1, 1, -0.5, -1]

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    teammateActions = self.receivedBroadcast
    # Process your teammate's broadcast!
    # Use it to pick a better action for yourself
    if self.receivedBroadcast and len(self.receivedBroadcast)>0:
        print(len(self.receivedBroadcast))
        pass

    actions = gameState.getLegalActions(self.index)

    filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), gameState, self.index)

    #currentAction = random.choice(filteredActions) # Change this!
    currentAction = self.actionHelper(gameState)
    return currentAction

  def getLimitedActions(self,state,index):
      actions = state.getLegalActions(index)
      filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), state, index)
      return filteredActions

  def actionHelper(self, state):
    actions = self.getLimitedActions(state, self.index)

    val = float('-inf')
    best = None
    for action in actions:
        new_state = state.generateSuccessor(self.index, action)
        new_state_val = self.evaluationFunction(new_state)

        if new_state_val > val:
            val = new_state_val
            best = action

    return best

  def evaluationFunction(self, state):
    foods = state.getFood().asList()
    ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
    friends = [state.getAgentPosition(pacman) for pacman in state.getPacmanTeamIndices() if pacman != self.index]

    pacman = state.getAgentPosition(self.index)

    closestFood = min(self.distancer.getDistance(pacman, food) for food in foods) + 2.0 \
        if len(foods) > 0 else 1.0
    closestGhost = min(self.distancer.getDistance(pacman, ghost) for ghost in ghosts) + 1.0 \
        if len(ghosts) > 0 else 1.0
    closestFriend = min(self.distancer.getDistance(pacman, friend) for friend in friends) + 1.0 \
        if len(friends) > 0 else 1.0

    closestFoodReward = 1.0 / closestFood
    closestGhostPenalty = 1.0 / (closestGhost ** 2) if closestGhost < 20 else 0
    closestFriendPenalty = 1.0 / (closestFriend ** 2) if closestFriend < 5 else 0

    numFood = len(foods)

    features = [-numFood, closestFoodReward, closestGhostPenalty, closestFriendPenalty]

    value = sum(feature * weight for feature, weight in zip(features, self.weights))
    return value

def actionsWithoutStop(legalActions):
  """
  Filters actions by removing the STOP action
  """
  legalActions = list(legalActions)
  if Directions.STOP in legalActions:
    legalActions.remove(Directions.STOP)
  return legalActions

def actionsWithoutReverse(legalActions, gameState, agentIndex):
  """
  Filters actions by removing REVERSE, i.e. the opposite action to the previous one
  """
  legalActions = list(legalActions)
  reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
  if len (legalActions) > 1 and reverse in legalActions:
    legalActions.remove(reverse)
  return legalActions
