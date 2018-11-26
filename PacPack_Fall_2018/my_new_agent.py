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

dir = [(-1,0),(1,0),(0,-1),(0,1)]
#for i in reversed(range(-1, 2)):
#    for j in range(-1, 2):
#        dir.append((j, i))

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


class MyAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.plan=[]
        self.followPlanFlag=True
        self.weights = Counter()
        self.weights['numFood'] = 1
        self.weights['closestFoodReward'] = 1
        self.weights['closestGhostPenalty'] = -10
        self.weights['closestFriendPenalty'] = -1
        self.features = Counter()

        self.walls = gameState.getWalls()
        self.foodGrid = gameState.getFood()
        self.dangerFood = list()
        self.foodWithEnv = list()
        for food in gameState.getFood().asList():
            wallsAround = 0
            for offset in dir:
                x = food[0] + offset[0]
                y = food[1] + offset[1]
                if self.walls[x][y]:
                    wallsAround += 1
            if wallsAround >= 3: self.dangerFood.append(food)
            self.foodWithEnv.append((food, wallsAround))
        self.dangerFood = set(self.dangerFood)

        self.dangerRegion=[]
        for food in self.dangerFood:
            tempPos=food
            regionFlag=True
            action=0
            while regionFlag:
                successors=self.getSearchSuccessorsWithoutReverse(tempPos,gameState,action)
                if len(successors)==1:
                    print(tempPos,action)
                    self.dangerRegion.append(tempPos)
                    tempPos=successors[0][0]
                    action=successors[0][1]
                else: regionFlag=False

        #nearbyDangerFood = set()
        #for food in self.dangerFood:
        #    for offset in dir:
        #        x = offset[0] + food[0]
        #        y = offset[1] + food[1]
        #        if self.foodGrid[x][y]:
        #            nearbyDangerFood.add((x, y))
        #self.dangerFood = list(self.dangerFood)
        #self.dangerFood.extend(list(nearbyDangerFood))
        #self.dangerFood = set(self.dangerFood)
        for food in self.dangerFood:
            self.debugDraw(food, [0,0,1])
        for rigon in self.dangerRegion:
            self.debugDraw(rigon, [0,1,0])

        self.friendIsStupid = True

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
        if not self.followPlanFlag:
            print("agent reflex")
        #print(self.followPlanFlag)
        #plan
        if self.followPlanFlag==True:
            pacman = gameState.getAgentPosition(self.index)
            if self.plan==[] or pacman==self.start:
                self.plan=self.PlanFunction(gameState)
                #print("plan:",self.plan)
            if len(self.plan)!=0:
                currentAction=self.plan.pop()
        ####reflex
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
    #def lureModeAction(self,state):
    #    actions=state.getLegalActions(self.index)

    def PlanFunction(self, state):
        foods = state.getFood().asList()
        ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
        friends = [state.getAgentPosition(pacman) for pacman in state.getPacmanTeamIndices() if pacman != self.index]
        pacman = state.getAgentPosition(self.index)

        #Remove friend's target
        friendPos=friends[0]
        ghostPos=ghosts[0]
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
        return self.aStarSearch(state,pacman,closestFood,ghostPos)

    def aStarSearch(self,state, pos,des,ghost):
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
                for (successorPos, action, stepCost) in self.getSearchSuccessors(nodePos,state,ghost,True):
                    nextPos=successorPos
                    #nextActionList=nodeActionList.copy()
                    nextActionList=[]
                    if len(nodeActionList)!=0:
                        for i in range(len(nodeActionList)):
                            nextActionList.append(nodeActionList[i])
                    nextActionList.insert(0,action)
                    #nextActionList.append(action)
                    nextCost=nodeCost+stepCost
                    fringe.push([nextPos,nextActionList,nextCost],nextCost+self.heuristic(nextPos,des))
        else: return []

    def getSearchSuccessors(self,pos,state,ghost,ghostFlag=False):
        successors=[]
        for action in [Directions.NORTH,Directions.SOUTH,Directions.EAST,Directions.WEST]:
            posX,posY=pos
            if action==Directions.NORTH: posY+=1
            if action==Directions.SOUTH: posY-=1
            if action==Directions.EAST: posX+=1
            if action==Directions.WEST: posX-=1
            if posY>0 and posY<18 and posX>0 and posX<34:
                if not state.hasWall(posX,posY):
                    if not ghostFlag:
                        successors.append(((posX,posY),action,1))
                    elif util.manhattanDistance((posX,posY),ghost)>2:
                        successors.append(((posX,posY),action,1))
        return successors

    def getSearchSuccessorsWithoutReverse(self,pos,state,prevAction):
        successors=[]
        actions=[Directions.NORTH,Directions.SOUTH,Directions.EAST,Directions.WEST]
        if prevAction in actions:
            actions.remove(Directions.REVERSE[prevAction])
        for action in actions:
            posX,posY=pos
            if action==Directions.NORTH: posY+=1
            if action==Directions.SOUTH: posY-=1
            if action==Directions.EAST: posX+=1
            if action==Directions.WEST: posX-=1
            if posY>0 and posY<18 and posX>0 and posX<34:
                if not state.hasWall(posX,posY):
                    successors.append(((posX,posY),action,1))
        return successors

    def heuristic(self,pos,des):
        return util.manhattanDistance(pos,des)

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
