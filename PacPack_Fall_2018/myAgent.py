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
from util import Counter,PriorityQueue,Queue
import game
from util import nearestPoint
#import numpy as np


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


class ReinforcementAgent(BaseAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self,gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls()

        self.readWeights()
        self.prefixed_weight = Counter()
        self.prefixed_weight['bias'] = 0
        self.prefixed_weight['reverse'] = -1000
        self.prefixed_weight['numFood'] = 60
        self.prefixed_weight['closestFood'] = 5
        self.prefixed_weight['closestFriend'] = -0.5
        self.prefixed_weight['closestGhost'] = -1
        self.prefixed_weight['pacInTunnel'] = -0.07
        self.prefixed_weight['5*5space'] = 1
        self.prefixed_weight['2_step_foods']=1
        self.prefixed_weight["eatFoodFeature"]=5

        self.lastNeighbor = []
        self.lastFeature = Counter()

        self.alpha = 0.01
        self.gamma = 0.95
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
        new_state = gameState.generateSuccessor(self.index, action)
        feats = Counter()
        feats['bias'] = 0
        if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
            feats['reverse'] = 1
        else: feats['reverse'] = 0

        foods = new_state.getFood().asList()
        ghosts = [new_state.getAgentPosition(ghost) for ghost in new_state.getGhostTeamIndices()]
        friends = [new_state.getAgentPosition(pacman) for pacman in new_state.getPacmanTeamIndices() if pacman != self.index]
        pacman = new_state.getAgentPosition(self.index)
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
        feats['numFood'] = -numFood/60
        feats['closestFood'] = closestFoodReward
        feats['closestFriend'] = closestFriendPenalty
        feats['closestGhost'] = closestGhostPenalty

        pacActions = new_state.getLegalActions(new_state.getGhostTeamIndices()[0])
        pacActions = actionsWithoutStop(pacActions)
        feats['pacInTunnel'] = 0
        if len(pacActions) == 2:
            feats['pacInTunnel'] = 1

        futureMoves = 0
        neighbour_foods=0
        freelist=[]
        for a in new_state.getLegalActions(self.index):
            s = new_state.generateSuccessor(self.index, a)
            new_actions=s.getLegalActions(self.index)
            for i in range(len(new_actions)):
                new_s=s.generateSuccessor(self.index, new_actions[i])
                new_pos=new_s.getAgentPosition(self.index)
                if  new_pos not in freelist:
                    if s.hasFood(new_pos[0],new_pos[1]):
                        neighbour_foods+=1
                    freelist.append(new_pos)
            #futureMoves += 1 + len(s.getLegalActions(self.index))
        #print(freelist,len(freelist))
        #print("neighbour_foods",neighbour_foods)
        feats['5*5space'] = len(freelist)/ 13
        feats["2_step_foods"]=neighbour_foods

        posX,posY=new_state.getAgentPosition(self.index)
        if gameState.hasFood(posX,posY):
            feats["eatFoodFeature"]=1
        #pacX, pacY = new_state.getAgentPosition(self.index)
        #freespace=0
        #walls=new_state.getWalls()
        #xlb=max(pacX-2,0)
        #xub=min(pacX+2,walls.width-1)
        #for i in range(xlb,xub+1):
        #    jrange=5-2*abs(pacX-i)
        #    jlb=max(pacY-jrange,0)
        #    jub=min(pacY+jrange,walls.height-1)
        #    #print("walls:",walls.width,walls.height)
        #    for j in range(jlb,jub+1):
        #        if new_state.hasWall(i,j)==False:
        #            freespace+=1
        #print("freespace",freespace)
        #feats['5*5space'] = freespace / 13


        return feats

    def getQValue(self, state, action,fixed=0):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #print(self.weights)
        #print(self.featExtractor.getFeatures(state,action))
        if fixed==0:
            return self.weight*self.getFeatures(state,action)
        else:
            return self.prefixed_weight*self.getFeatures(state,action)

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

            Qvalues = [self.getQValue(state,action,1) for action in actions]
            best_Qvalue=max(Qvalues)
            #print("bestQvalue",best_Qvalue)
            best_indices=[index for index in range(len(Qvalues)) if Qvalues[index]==best_Qvalue]
            #print("actions")
            #for i in range(len(actions)):
            #    print(actions[i],"Q value:",Qvalues[i])
            #print("best_actions",best_indices,"  size:",len(best_indices))
            chosen_index=random.choice(best_indices)
            best_action=actions[chosen_index]
            #print("best action:",best_action)
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
        #print("action choosen",action)
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
            #print('weight:',self.weight," lastfeature",self.lastFeature)
            #print(" lastfeature",self.lastFeature)
            difference=reward+self.gamma*self.computeValueFromQValues(state)-self.weight*self.lastFeature
            #print("reward:",reward)
            #print("difference:",difference)
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
        with open("epsilon.txt", "r")as f:
            self.epsilon = float(f.read())

class TutoredRLAgent(ReinforcementAgent):
    def update(self, state):
        ReinforcementAgent.update(self, state)
        if self.receivedBroadcast != None:
            features, rewards, lastFeatures = self.receivedBroadcast
            difference = rewards + self.gamma * features - self.weight * lastFeatures
            for feature in lastFeatures:
                self.weight[feature] += self.alpha / 2 * difference * self.lastFeature[feature]



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


class MyAgent(BaseAgent):
    def registerInitialState(self, gameState):
        BaseAgent.registerInitialState(self, gameState)
        self.plan=[]
        self.followPlanFlag=True
        self.weights = Counter()
        self.weights['numFood'] = 1
        self.weights['closestFoodReward'] = 1
        self.weights['closestGhostPenalty'] = -10
        self.weights['closestFriendPenalty'] = -1
        self.features = Counter()

    def getFeatures(self, state):
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

        #features=Counter()
        self.features['numFood']=-numFood
        self.features['closestFoodReward']=closestFoodReward
        self.features['closestGhostPenalty']=closestGhostPenalty
        self.features['closestFriendPenalty']=closestFriendPenalty

        return self.features

    def evaluationFunction(self,state):
        self.getFeatures(state)
        return self.features*self.weights

    def chooseAction(self, gameState):
        currentAction = self.actionHelper(gameState)
        self.detectDanger(gameState)
        #print(self.followPlanFlag)
        #plan
        if self.followPlanFlag==True:
            pacman = gameState.getAgentPosition(self.index)
            if self.plan==[] or pacman==(1,2) or pacman==(1,1):
                self.plan=self.PlanFunction(gameState)
            currentAction=self.plan.pop()
        ##reflex
        else:
            self.plan=[]
            currentAction = self.actionHelper(gameState)
        #print(currentAction)
        return currentAction

    def actionHelper(self, state):
        #actions = self.getLimitedActions(state, self.index)
        actions = state.getLegalActions(self.index)

        val = float('-inf')
        best = None
        for action in actions:
            new_state = state.generateSuccessor(self.index, action)
            new_state_val = self.evaluationFunction(new_state)

            if new_state_val > val:
                val = new_state_val
                best = action

        return best

    def detectDanger(self,state):
        ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
        ghost=ghosts[0]
        pacman = state.getAgentPosition(self.index)
        distance=self.distancer.getDistance(pacman,ghost)
        self.followPlanFlag=False
        if distance<8:
            if util.flipCoin(distance/8):
                self.followPlanFlag=True
        else:
            self.followPlanFlag=True

    #to be continued
    def lureModeAction(self,state):
        actions=state.getLegalActions(self.index)

    def PlanFunction(self, state):
        foods = state.getFood().asList()
        ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
        friends = [state.getAgentPosition(pacman) for pacman in state.getPacmanTeamIndices() if pacman != self.index]
        pacman = state.getAgentPosition(self.index)

        #Remove friend's target
        friendPos=friends[0]
        foods = state.getFood().asList()
        numFood = len(foods)
        friendTargets = [food for food in foods if self.distancer.getDistance(friendPos, food) < 7]
        targetsNearTargets = []
        for f in friendTargets:
            targetsNearTargets.extend([food for food in foods if self.distancer.getDistance(f, food) < 4])
        friendTargets.extend(targetsNearTargets)
        friendTargets = set(friendTargets)
        if numFood - len(friendTargets) > 3:
            for f in friendTargets:
                if f in foods: foods.remove(f)


        closestFood=pacman
        if len(foods) > 0:
            closestFoodDist = min(self.distancer.getDistance(pacman, food) for food in foods)
            closestFoods=[food for food in foods if self.distancer.getDistance(pacman,food)==closestFoodDist]
            closestFood=random.choice(closestFoods)
        ##A* Search
        return self.aStarSearch(state,pacman,closestFood)

    def aStarSearch(self,state, pos,des):
        """Search the node that has the lowest combined cost and heuristic first."""
        "*** YOUR CODE HERE ***"
        fringe=util.PriorityQueue()
        fringe.push([pos,[],0],0)
        closed_set=[]
        #print('pos',pos,' des',des)
        while not fringe.isEmpty():
            #print('now the fringe have',len(fringe.heap))
            #print('now the closed set is',closed_set)
            nodePos,nodeActionList,nodeCost=fringe.pop()
            #print("poped pos",nodePos)
            if nodePos==des:
                return nodeActionList
            if not nodePos in closed_set:
                closed_set.append(nodePos)
                for (successorPos, action, stepCost) in getSearchSuccessors(nodePos,state):
                    nextPos=successorPos
                    nextActionList=nodeActionList.copy()
                    nextActionList.insert(0,action)
                    #nextActionList.append(action)
                    nextCost=nodeCost+stepCost
                    fringe.push([nextPos,nextActionList,nextCost],nextCost+heuristic(nextPos,des))
        else: return []


def getSearchSuccessors(pos,state):
    successors=[]
    for action in [Directions.NORTH,Directions.SOUTH,Directions.EAST,Directions.WEST]:
        posX,posY=pos
        if action==Directions.NORTH: posY+=1
        if action==Directions.SOUTH: posY-=1
        if action==Directions.EAST: posX+=1
        if action==Directions.WEST: posX-=1
        if posY>0 and posY<18 and posX>0 and posX<34:
            if not state.hasWall(posX,posY):
                successors.append(((posX,posY),action,1))
    return successors
def heuristic(pos,des):
    return util.manhattanDistance(pos,des)
