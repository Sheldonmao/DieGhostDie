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
from game import Directions, Actions
from util import Counter
import game
from util import nearestPoint


#########
# Agent #
#########
class BaseAgent(CaptureAgent):
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
        print(gameState.data.time)
        print(len(gameState.getFood().asList()))
        teammateActions = self.receivedBroadcast
        # Process your teammate's broadcast!
        # Use it to pick a better action for yourself

        actions = gameState.getLegalActions(self.index)

        filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), gameState, self.index)

        # currentAction = random.choice(filteredActions) # Change this!
        currentAction = self.actionHelper(gameState)
        return currentAction

    def getLimitedActions(self, state, index):
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

        myState = state.getAgentState(self.index)

        value = sum(feature * weight for feature, weight in zip(features, self.weights))
        return value

class MyAgent(BaseAgent):
    def toString(self):
        return "MyAgent"

class ReinforcementAgent(BaseAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self,gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls()

        self.readWeights()
        self.lastNeighbor = []
        self.lastFeature = Counter()

        self.alpha = 0.001
        self.gamma = 0.8
        self.epsilon = 0.2
        self.reverse_prob=0.5

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
        feats['numFood'] = numFood
        feats['closestFood'] = closestFoodReward
        feats['closestFriend'] = closestFriendPenalty
        feats['closestGhost'] = closestGhostPenalty

        pacActions = gameState.getLegalActions(gameState.getGhostTeamIndices()[0])
        pacActions = actionsWithoutStop(pacActions)
        feats['pacInTunnel'] = 0
        if len(pacActions) == 2:
            feats['pacInTunnel'] = 1

        futureMoves = 0
        for a in gameState.getLegalActions(self.index):
            s = gameState.generateSuccessor(self.index, a)
            futureMoves += 1 + len(s.getLegalActions(self.index))
        feats['5*5space'] = futureMoves / 6
        return feats

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #print(self.weights)
        #print(self.featExtractor.getFeatures(state,action))
        return self.weight*self.getFeatures(state,action)

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions=state.getLegalActions(self.index)
        actions = actionsWithoutStop(actions)
        best_Qvalue=0
        if len(actions)!=0:
            Qvalues = [self.getQValue(state,action) for action in actions]
            best_Qvalue=max(Qvalues)
        return best_Qvalue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions=state.getLegalActions(self.index)
        actions = actionsWithoutStop(actions)
        best_action=None
        if len(actions)!=0:
            #print("actions",actions,"  size:",len(actions))
            Qvalues = [self.getQValue(state,action) for action in actions]
            best_Qvalue=max(Qvalues)
            #print("bestQvalue",best_Qvalue)
            best_indices=[index for index in range(len(Qvalues)) if Qvalues[index]==best_Qvalue]
            #print("best_indicies",best_indices,"  size:",len(best_indices))
            chosen_index=random.choice(best_indices)
            best_action=actions[chosen_index]
        return best_action

    def chooseAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        pacX, pacY = state.getAgentPosition(self.index)
        # Pick Action
        legalActions = state.getLegalActions(self.index)
        if util.flipCoin(self.reverse_prob):
            legalActions=actionsWithoutReverse(legalActions,state,self.index)
                        
        action = None
        "*** YOUR CODE HERE ***"
        if pacX <= 2 * self.bornHardLevel - 1:
            if ((pacX + 1) / 2) % 2 == 1:
                action = Directions.NORTH if Directions.NORTH in legalActions else None
            else:
                action = Directions.SOUTH if Directions.SOUTH in legalActions else None
        if len(legalActions)!=0 and action == None:
            if util.flipCoin(self.epsilon):
                action=random.choice(legalActions)
            else:
                action=self.computeActionFromQValues(state)
        self.update(state)
        self.lastFeature=self.getFeatures(state,action)
        self.saveWeights(state)
        return action

    def update(self, state):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        reward=self.getReward(state)
        if len(self.lastFeature.keys()) != 0:
            print('weight:',self.weight," lastfeature",self.lastFeature)
            difference=reward+self.gamma*self.computeValueFromQValues(state)-self.weight*self.lastFeature
            print("difference:",difference)
            for feature in self.lastFeature:
                self.weight[feature]+=self.alpha*difference*self.lastFeature[feature]


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
            if food[x][y] == True:
                self.lastNeighbor.append((x, y))

        return eatenPenalty + eatFoodReward + foodDecreaseReward

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getWeights(self):
        return self.weight

    def saveWeights(self, gameState):
        if len(gameState.getFood().asList()) or gameState.data.time >= 1190:
            f = open("train.txt", "w")
            for key in self.weight.keys():
                value = str(self.weight[key])
                toPrint = key + ':' + value
                print(toPrint, file=f)

    def readWeights(self):
        self.weight = Counter()
        with open("train.txt", "r") as f:
            for line in f.readlines():
                content = line.split(':')
                self.weight[content[0]] = float(content[1])



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
    if len(legalActions) > 1 and reverse in legalActions:
        legalActions.remove(reverse)
    return legalActions
