# staffBot.py
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
# This file was based on the starter code for student bots, and edited 
# by Mesut (Xiaocheng) Yang and Roshan Rao


from captureAgents import CaptureAgent
from game import Directions, Actions
from util import Counter

#########
# Agent #
#########


class SimpleStaffBot(CaptureAgent):
    """
    A Simple agent to serve as an example of the necessary agent structure.
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
        self.weights = [1, 1, 0, 0]

    def getLimitedActions(self, state, index, remove_reverse=True):
        """
        Limit the actions, removing 'Stop', and the reverse action if possible.
        """
        actions = state.getLegalActions(index)
        actions.remove('Stop')

        if len(actions) > 1 and remove_reverse:
            rev = Directions.REVERSE[state.getAgentState(index).configuration.direction]
            if rev in actions:
                actions.remove(rev)

        return actions

    def chooseAction(self, gameState):
        """
        Reflex agent that follows its plan.
        """

        # Follow plan if available and possible
        if self.toBroadcast and len(self.toBroadcast) > 0:
            action = self.toBroadcast.pop(0)
            if action in gameState.getLegalActions(self.index):
                ghosts = [gameState.getAgentPosition(ghost) for ghost in gameState.getGhostTeamIndices()]

                pacman = gameState.getAgentPosition(self.index)
                closestGhost = min(self.distancer.getDistance(pacman, ghost) for ghost in ghosts) \
                    if len(ghosts) > 0 else 1.0
                # If the ghost is nearby, replan
                if closestGhost >= 10:
                    return action

        # use actionHelper to pick an action
        currentAction = self.actionHelper(gameState)
        futureActions = self.generatePlan(gameState.generateSuccessor(self.index, currentAction), 3)

        self.toBroadcast = futureActions
        return currentAction

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

    def generatePlan(self, state, plan_length):
        plan = []
        other_index = state.getPacmanTeamIndices()
        other_index.remove(self.index)
        other_index = other_index[0]
        for i in range(plan_length):
            # If agent doesn't get a broadcast or the broadcasted move is illegal, ignore the move
            if self.receivedBroadcast and len(self.receivedBroadcast) > i:
                action = self.receivedBroadcast[i]
                if action in state.getLegalActions(other_index):
                    state = state.generateSuccessor(other_index, action)
                else:
                    # NOTE: If you're worried about potentially broadcasting illegal actions, uncomment this line.
                    # You should only broadcast an illegal action around the time your agent dies (because the death is unexpected).
                    # In all other circumstances, your broadcasted actions should be legal.

                    print('You broadcasted an illegal action!')
                    pass  

            action = self.actionHelper(state)
            plan.append(action)
            state = state.generateSuccessor(self.index, action)
        return plan

class TutorBot(SimpleStaffBot):
    def registerInitialState(self, gameState):
        SimpleStaffBot.registerInitialState(self, gameState)
        self.lastNeighbor = []
        self.lastFeature = Counter()
        self.start = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls()
        self.bornHardLevel = 0
        for i in range(1, 4):
            col = 2 * i
            walls = 0
            for j in range(1, self.walls.height - 1):
                if self.walls[col][j] == True:
                    walls += 1
            if walls < self.walls.height - 3:
                break
            self.bornHardLevel += 1
        self.thisAction = None

    def getReward(self, gameState):
        eatenPenalty = 0
        if gameState.getAgentPosition(self.index) == self.start:
            eatenPenalty = -10 * self.bornHardLevel
        eatFoodReward = 0
        pacman = gameState.getAgentPosition(self.index)

        if len(self.lastNeighbor) != 0:
            x, y = pacman
            if (x, y) in self.lastNeighbor:
                eatFoodReward = 10

        foodDecreaseReward = 0
        if len(self.observationHistory) != 0:
            lastFoodNum = len(self.observationHistory[-1].getFood().asList())
            presentFoodNum = len(gameState.getFood().asList())
            foodDecreaseReward = 10 * (lastFoodNum - presentFoodNum)

        # 储存我的四邻域食物情况
        self.lastNeighbor = []
        food = gameState.getFood()
        legalNeighbors = Actions.getLegalNeighbors(pacman, self.walls)
        for i in range(len(legalNeighbors)):
            x, y = legalNeighbors[i]
            if food[x][y]:
                self.lastNeighbor.append((x, y))

        return eatenPenalty + eatFoodReward + foodDecreaseReward

    def getFeatures(self, gameState, action):
        feats = Counter()
        feats['bias'] = 1
        if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
            feats['reverse'] = 1
        else: feats['reverse'] = 0

        foods = gameState.getFood().asList()
        ghosts = [gameState.getAgentPosition(ghost) for ghost in gameState.getGhostTeamIndices()]
        friends = [gameState.getAgentPosition(pacman) for pacman in gameState.getPacmanTeamIndices() if pacman != self.index]
        pacman = gameState.getAgentPosition(self.index)
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
        feats['numFood'] = numFood/60
        feats['closestFood'] = closestFoodReward
        feats['closestFriend'] = closestFriendPenalty
        feats['closestGhost'] = closestGhostPenalty

        pacActions = gameState.getLegalActions(gameState.getGhostTeamIndices()[0])
        pacActions = actionsWithoutStop(pacActions)
        feats['pacInTunnel'] = 0
        if len(pacActions) == 2:
            feats['pacInTunnel'] = 1
        return feats

    def chooseAction(self, gameState):
        self.thisAction = SimpleStaffBot.chooseAction(self, gameState)
        return self.thisAction

    def generatePlan(self, state, plan_length):
        if self.thisAction == None:
            return None
        features = self.getFeatures(state, self.thisAction)
        rewards = self.getReward(state)
        if len(self.lastFeature.keys()) == 0:
            rtn = [features, rewards, features]
        else: rtn = [features, rewards, self.lastFeature]
        self.lastFeature = features
        return rtn


def actionsWithoutStop(legalActions):
    """
    Filters actions by removing the STOP action
    """
    legalActions = list(legalActions)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    return legalActions
