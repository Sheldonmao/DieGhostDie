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
from game import Grid
#import numpy as np

dir = [(-1,0),(1,0),(0,-1),(0,1)]

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
        #print(gameState.data.time)
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
        self.forceFlag=False
        self.followPlanFlag=True
        self.replanFlag=True
        self.weights = Counter()
        self.weights['numFood'] = 1
        self.weights['closestFoodReward'] = 1
        self.weights['closestGhostPenalty'] = -1
        self.weights['closestFriendPenalty'] = -0.5
        self.weights['dangerGridPenalty']=-1
        self.weights['subDangerGridPenalty']=-1
        self.features = Counter()

        self.targetFood=(0,0)
        self.walls = gameState.getWalls()
        self.foodGrid = gameState.getFood()
        self.dangerFood = list()
        for food in gameState.getFood().asList():
            wallsAround = 0
            for offset in dir:
                x = food[0] + offset[0]
                y = food[1] + offset[1]
                if self.walls[x][y]:
                    wallsAround += 1
            if wallsAround >= 3: self.dangerFood.append(food)
        self.dangerFood = set(self.dangerFood)

        [self.dangerFoodRegion,self.dangerFoodRegionList]=self.regionGrowing(self.dangerFood,gameState)

        self.foodExitGrid=Grid(self.foodGrid.width, self.foodGrid.height)
        for regionlist in self.dangerFoodRegionList:
            pos=regionlist[-1]
            self.foodExitGrid[pos[0]][pos[1]]=True
            #self.debugDraw(pos,[0,0,1])

        self.deadEnds=[]
        self.narrows=[]
        for i in range(1,self.foodGrid.width-1):
            for j in range(1,self.foodGrid.height-1):
                if not self.walls[i][j]:
                    wallsAround = 0
                    for offset in dir:
                        x = i + offset[0]
                        y = j + offset[1]
                        if self.walls[x][y]:
                            wallsAround += 1
                    if wallsAround==2:
                        self.narrows.append((i,j))
                    if wallsAround >= 3:
                        self.deadEnds.append((i,j))
                        #self.debugDraw((i,j),[1,0,0])
        [self.dangerRegion,self.dangerRegionList]=self.regionGrowing(self.deadEnds,gameState)
        self.dangerGrid = Grid(self.foodGrid.width, self.foodGrid.height)
        for pos in self.dangerRegion:
            self.dangerGrid[pos[0]][pos[1]]=True
            self.debugDraw(pos,[0,1,0])

        self.dangerExitList=[]
        self.subDangerRegion=[]
        for posrange in self.dangerRegionList:
            self.dangerExitList.append(posrange[-1])
        for i in range(len(self.dangerExitList)-1):
            for j in range(i+1,len(self.dangerExitList)):
                pos=self.dangerExitList[i]
                des=self.dangerExitList[j]
                if self.distancer.getDistance(pos,des)<=2:
                    middle=(int((pos[0]+des[0])/2),int((pos[1]+des[1])/2))
                    if self.walls[middle[0]][middle[1]]==True:
                        middle=(middle[0]+1,middle[1]+1)
                    if self.walls[middle[0]][middle[1]]==False:
                        self.debugDraw(middle,[0,0,1])
                        self.subDangerRegion.append(middle)

        #for food in self.dangerFood:
        #     self.debugDraw(food, [0,0,1])
        # for rigon in self.dangerFoodRegion:
        #     self.debugDraw(rigon, [0,1,0])
        # for region in self.dangerFoodRegionList[0]:
        #     self.debugDraw(region,[1,0,0])
        self.closedTargets=[]
        self.hesitate=0
        self.friendIsStupid = True

    def regionGrowing(self,seeds,state):
        Region=[]
        RegionList=[]
        for food in seeds:
            templist=[]
            tempPos=food
            regionFlag=True
            action=0
            while regionFlag:
                successors=self.getSearchSuccessorsWithoutReverse(tempPos,state,action)
                if len(successors)==1:
                    #print(tempPos,action)
                    Region.append(tempPos)
                    templist.append(tempPos)
                    tempPos=successors[0][0]
                    action=successors[0][1]
                else:
                    regionFlag=False
                    RegionList.append(templist)
        return [Region,RegionList]

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

        dangergrid=0
        if self.dangerGrid[pacman[0]][pacman[1]]==True:
            dangergrid=1

        subDangerGrid=0
        if pacman in self.subDangerRegion:
            subDangerGrid=1

        #features=Counter()
        self.features['numFood']=-numFood
        self.features['closestFoodReward']=closestFoodReward
        self.features['closestGhostPenalty']=closestGhostPenalty
        self.features['closestFriendPenalty']=closestFriendPenalty
        self.features['dangerGridPenalty']=dangergrid
        self.features['subDangerGridPenalty']=subDangerGrid

        return self.features

    def evaluationFunction(self,state):
        self.getFeatures(state)
        return self.features*self.weights

    def packPlan(self,state):
        pacman=state.getAgentPosition(self.index)
        if self.foodExitGrid[pacman[0]][pacman[1]]==True:
            self.foodExitGrid[pacman[0]][pacman[1]]=False
            self.plan=[]
            for i in range(len(self.dangerFoodRegionList)):
                if pacman==self.dangerFoodRegionList[i][-1]:
                    self.plan=self.PosList2ActionList(self.dangerFoodRegionList[i])
            if len(self.plan)!=0:
                #print("pack plan",self.plan)
                return True
        return self.forceFlag

    def updateEatenPack(self,state):
        pacman=state.getAgentPosition(self.index)
        friends = [state.getAgentPosition(pacman) for pacman in state.getPacmanTeamIndices() if pacman != self.index]
        friend=friends[0]
        positions=[pacman,friend]
        for pos in positions:
            index=-1
            if pos in self.dangerFood:
                for i in range(len(self.dangerFoodRegionList)):
                    if pos==self.dangerFoodRegionList[i][0]:
                        index=i
            if index!=-1:
                for temppos in self.dangerFoodRegionList[index]:
                    self.debugDraw(temppos,[1,0,0])
                self.dangerFoodRegionList.remove(self.dangerFoodRegionList[index])
                if pos==pacman:
                    self.forceFlag=False
                    self.debugDraw(pos,[1,1,1])
                    print("goal",pos,self.forceFlag)

    def chooseAction(self, gameState):
        currentAction = self.actionHelper(gameState)
        self.updateEatenPack(gameState)
        self.forceFlag=self.packPlan(gameState)
        self.detectDanger(gameState)
        computeFlag=False
        #if not self.followPlanFlag:
        #    print("not follow plan Flag",self.forceFlag)
        #plan
        if self.followPlanFlag==True or self.forceFlag==True:
            if self.replanFlag==True and self.forceFlag==False:
                self.plan=[]
                self.replanFlag=False
            pacman = gameState.getAgentPosition(self.index)
            if self.plan==[] or pacman==self.start:
                self.plan=self.PlanFunction(gameState)

                #draw plan
                nextPos=pacman
                b=random.random()
                for action in reversed(self.plan):
                    nextPos=self.stepAction(nextPos,action)
                    self.debugDraw(nextPos,[0.5,0.5,b])

                computeFlag=True
            if computeFlag==False:
                planB=self.PlanFunction(gameState)
                if len(planB)< len(self.plan)-self.hesitate*5:
                    self.hesitate+=1
                    if self.hesitate==10:
                        self.hesitate=0
                    self.plan=planB
                    nextPos=pacman
                    b=random.random()
                    for action in reversed(self.plan):
                        nextPos=self.stepAction(nextPos,action)
                        self.debugDraw(nextPos,[0.5,0.5,b])

            if len(self.plan)!=0:
                currentAction=self.plan.pop()
                self.debugDraw(pacman,[0,0,0])
        ####reflex
        else:
            print("reflex")
            self.plan=[]
            currentAction = self.actionHelper(gameState)
        #print(currentAction)
        return currentAction

    def stepAction(self,pos,action):
        x,y=pos
        if action==Directions.NORTH: return (x,y+1)
        if action==Directions.SOUTH: return (x,y-1)
        if action==Directions.EAST: return (x+1,y)
        if action==Directions.WEST: return (x-1,y)

    def actionHelper(self, state):
        #actions = self.getLimitedActions(state, self.index)
        pacman=state.getAgentPosition(self.index)
        ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
        ghost=ghosts[0]
        ghostDist=util.manhattanDistance(pacman,ghost)
        if self.dangerGrid[pacman[0]][pacman[1]]==True and ghostDist>=3:
            index=(-1,-1)
            for i in range(len(self.dangerRegionList)):
                for j in range(len(self.dangerRegionList[i])-1):
                    if pacman==self.dangerRegionList[i][j]:
                        index=(i,j+1)
            if index!=(-1,-1):
                return self.chooseNeighbourActioin(pacman,self.dangerRegionList[index[0]][index[1]])
        actions = actionsWithoutStop(state.getLegalActions(self.index))
        val = float('-inf')
        best = None
        for action in actions:
            new_state = state.generateSuccessor(self.index, action)
            new_state_val = self.evaluationFunction(new_state)
            if new_state_val > val:
                val = new_state_val
                best = action
        return best

    ##reversed action list
    def PosList2ActionList(self,poslist):
        actionlist=[]
        for i in range(len(poslist)-1):
            actionlist.append(self.chooseNeighbourActioin(poslist[i+1],poslist[i]))
        return actionlist
    #be careful only neighour points can do this
    def chooseNeighbourActioin(self,pos,des):
        xdif=des[0]-pos[0]
        ydif=des[1]-pos[1]
        if xdif==-1:    return Directions.WEST
        if xdif==1:     return Directions.EAST
        if ydif==-1:    return Directions.SOUTH
        if ydif==1:     return Directions.NORTH

    def detectDanger(self,state):
        ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
        friends = [state.getAgentPosition(pacman) for pacman in state.getPacmanTeamIndices() if pacman != self.index]
        friend=friends[0]
        ghost=ghosts[0]
        pacman = state.getAgentPosition(self.index)
        distance=self.distancer.getDistance(pacman,ghost)
        self.followPlanFlag=False
        if distance<5:
            #if util.flipCoin(distance/5):
            #    self.followPlanFlag=True
            if distance<=2:
                self.followPlanFlag=False
        else:
            self.followPlanFlag=True

        foodGrid=state.getFood()
        if foodGrid[self.targetFood[0]][self.targetFood[1]]==False:
            self.replanFlag=True
        if self.distancer.getDistance(pacman,friend)<4:
            self.closedTargets.append(self.targetFood)
            self.replanFlag=True

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
        closestFriendFoodDist = min(self.distancer.getDistance(friendPos, food) for food in foods)
        friendTargets = [food for food in foods if self.distancer.getDistance(friendPos, food) < closestFriendFoodDist+1]
        targetsNearTargets = []
        for f in friendTargets:
            targetsNearTargets.extend([food for food in foods if self.distancer.getDistance(f, food) < 3])
        friendTargets.extend(targetsNearTargets)
        friendTargets = set(friendTargets)
        if numFood - len(friendTargets) > 3:
            for f in friendTargets:
                if f in foods: foods.remove(f)

        ghostTargets = [food for food in foods if self.distancer.getDistance(ghostPos, food) < 4]
        targetsNearTargets = []
        for f in ghostTargets:
            targetsNearTargets.extend([food for food in foods if self.distancer.getDistance(f, food) < 3])
        ghostTargets.extend(targetsNearTargets)
        ghostTargets = set(ghostTargets)
        if numFood - len(ghostTargets) > 3:
            for f in ghostTargets:
                if f in foods: foods.remove(f)

        print("len of closed targets",len(self.closedTargets))
        for target in self.closedTargets:
            if target in foods:
                foods.remove(target)
        dist=max(2-len(self.closedTargets),0)
        #### tends to right
        # if util.flipCoin(1-state.data.time/1200):
        #     removelist=[]
        #     for food in foods:
        #         if food[0]<17:
        #             removelist.append(food)
        #     for food in removelist:
        #         foods.remove(food)


        self.targetFood=pacman
        if len(foods) > 0:
            closestFoodDist = min(self.distancer.getDistance(pacman, food) for food in foods)
            closestFoods=[food for food in foods if self.distancer.getDistance(pacman,food)==closestFoodDist]
            self.targetFood=random.choice(closestFoods)
        #print("target",self.targetFood,pacman)
        ##A* Search
        return self.aStarSearch(state,pacman,self.targetFood,ghostPos,dist)


    def aStarSearch(self,state, pos,des,ghost,dist=2):
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
                self.closedTargets=[]
                return nodeActionList
            if not nodePos in closed_set:
                closed_set.append(nodePos)
                for (successorPos, action, stepCost) in self.getSearchSuccessors(nodePos,state,ghost,True,dist):
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
        else:
            self.closedTargets.append(des)
            return []

    def getSearchSuccessors(self,pos,state,ghost,ghostFlag=False,dist=2):
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
                    elif util.manhattanDistance((posX,posY),ghost)>=dist:
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
