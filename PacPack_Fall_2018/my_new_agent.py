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

RANDOM_ACTION_PROB = 0.4
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
        ### find index ###
        self.actionToTake = Directions.NORTH
        self.ghostIndex = gameState.getGhostTeamIndices()[0]
        self.friendIndex = -1
        for i in gameState.getPacmanTeamIndices():
            if i != self.index:
                self.friendIndex = i
                break
        self.ghostStart = gameState.getAgentPosition(self.ghostIndex)
        self.start = gameState.getAgentPosition(self.index)

        self.plan=[]
        self.forceFlag=False
        self.followPlanFlag=True
        self.replanFlag=True
        self.weights = Counter()
        self.weights['numFood'] = 1
        self.weights['closestFoodReward'] = 1
        self.weights['foodDangerRegionReward']=1
        self.weights['closestGhostPenalty'] = -1
        self.weights['closestFriendPenalty'] = -0.5
        self.weights['DeadEndGridPenalty']=-1
        self.weights['subDeadEndGridPenalty']=-1
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

        [_,self.dangerFoodRegionList]=self.regionGrowing(self.dangerFood,gameState)

        self.foodExitGrid=Grid(self.foodGrid.width, self.foodGrid.height)
        for regionlist in self.dangerFoodRegionList:
            pos=regionlist[-1]
            self.foodExitGrid[pos[0]][pos[1]]=True
            #self.debugDraw(pos,[0,0,1])

        self.bornHardLevel = 0
        for i in range(1, 4):
            col = 2 * i
            isWall = True
            for j in range(1, self.walls.height - 1):
                if i % 2 == 1:
                    if j != self.walls.height - 2 and not self.walls[col][j]:
                        isWall = False
                        break
                    elif j == self.walls.height - 2 and self.walls[col][j]:
                        isWall = False
                        break
                else:
                    if j != 1 and not self.walls[col][j]:
                        isWall = False
                        break
                    elif j == 1 and self.walls[col][j]:
                        isWall = False
                        break
            if isWall: self.bornHardLevel += 1

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
        [self.DeadEndRegion,self.DeadEndRegionList]=self.regionGrowing(self.deadEnds,gameState)
        self.DeadEndGrid = Grid(self.foodGrid.width, self.foodGrid.height)
        for pos in self.DeadEndRegion:
            self.DeadEndGrid[pos[0]][pos[1]]=True
            self.debugDraw(pos,[0,1,0])

        self.DangerSeeds=list(set(self.deadEnds).difference(set(self.dangerFood)))
        [_,self.DangerRegionList]=self.regionGrowing(self.DangerSeeds,gameState)

        for i in self.DangerRegionList:
            for j in i:
                self.debugDraw(j,[1,0,0])

        self.dangerExitList=[]
        self.subDeadEndRegion=[]
        for posrange in self.DeadEndRegionList:
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
                        self.subDeadEndRegion.append(middle)
        self.closedTargets=[]
        self.hesitate=0
        self.friendIsStupid = True

        ###Atributes for which evaluation to use for friend###
        self.friendIsStupid = False
        self.simpleEvalTimes = 0
        self.simpleRightTimes = 0
        self.lastGameState = None
        self.friendBehavior = Directions.STOP
        self.friendPrediction = [Directions.STOP, Directions.STOP]
        self.lastEvaluated = False
        self.toBroadcast = []

        ###redundant food groups, for gametree agent###
        self.foodGroup = [None]
        for food in self.dangerFood:
            thisGroup = set()
            regionSize = 0
            tempPos = food
            regionFlag = True
            action = 0
            while regionFlag:
                successors = self.getSearchSuccessorsWithoutReverse(tempPos, gameState, action)
                if len(successors) == 1:
                    print(tempPos, action)
                    self.dangerRegion.append(tempPos)
                    ###group food###
                    regionSize += 1
                    x, y = tempPos
                    if self.foodGrid[x][y]: thisGroup.add(tempPos)
                    tempPos = successors[0][0]
                    action = successors[0][1]
                else:
                    regionFlag = False
            self.foodGroup.append([thisGroup, regionSize + 1, tempPos])
        problematicFood = set()
        for i in range(1, len(self.foodGroup)):
            for f in self.foodGroup[i][0]: problematicFood.add(f)
        safeGroup = set([food for food in gameState.getFood().asList() if food not in problematicFood])
        self.foodGroup[0] = [safeGroup, 0, None]



    ##LOL: list of lists
    def lookUpList(self,LOL,Myelement):
        for i in range(len(LOL)):
            for j in range(len(LOL[i])):
                if Myelement==LOL[i][j]:
                    return [True,[i,j]]
        return [False,[]]

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

    ### not used ###
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

        DeadEndGrid=0
        #if self.DeadEndGrid[pacman[0]][pacman[1]]==True:
        if pacman in self.DeadEndRegion:
            DeadEndGrid=1

        subDeadEndGrid=0
        if pacman in self.subDeadEndRegion:
            subDeadEndGrid=1

        foodDangerRegionReward=0
        pacLookUp=self.lookUpList(self.dangerFoodRegionList,pacman)
        if pacLookUp[0]:
            foodDangerRegionReward=len(self.dangerFoodRegionList[pacLookUp[1][0]])

        #features=Counter()
        self.features['numFood']=-numFood
        self.features['closestFoodReward']=closestFoodReward
        self.features['foodDangerRegionReward']=foodDangerRegionReward
        self.features['closestGhostPenalty']=closestGhostPenalty
        self.features['closestFriendPenalty']=closestFriendPenalty
        self.features['DeadEndGridPenalty']=DeadEndGrid
        self.features['subDeadEndGridPenalty']=subDeadEndGrid

        return self.features

    ### not used ###
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
                self.DangerRegionList.append(self.dangerFoodRegionList[index])
                print('len of danger food region list',len(self.DangerRegionList))
                self.dangerFoodRegionList.remove(self.dangerFoodRegionList[index])
                if pos==pacman:
                    self.forceFlag=False
                    self.debugDraw(pos,[1,1,1])
                    #print("goal",pos,self.forceFlag)

    def chooseAction(self, gameState):
        self.toBroadcast = [] #initialize toBroadcast
        currentAction = self.actionHelper(gameState)
        self.updateFoodGroup(gameState)#update for gametree-version
        self.updateEatenPack(gameState)
        self.forceFlag=self.packPlan(gameState)
        self.detectDanger(gameState)
        computeFlag=False
        ###Variables for hard code###
        pacX, pacY = gameState.getAgentPosition(self.index)
        ghostPos = gameState.getAgentPosition(self.ghostIndex)
        friendPos = gameState.getAgentPosition(self.friendIndex)
        ### Hard Code 1 : lure the ghost back to home, when: ###
        ### near ghost's home and ghost near me and friend near food ###
        if pacX >= 32 - self.bornHardLevel * 2 \
                and self.distancer.getDistance((pacX, pacY), ghostPos) < 7:
            candidateGroups = [group for group in self.foodGroup if group[2] and len(group[0]) > 1]
            if self.distancer.getDistance((pacX, pacY), self.ghostStart) \
                    < self.distancer.getDistance(ghostPos, self.ghostStart) \
                    and len(candidateGroups) > 0\
                    and min([self.distancer.getDistance(gameState.getAgentPosition(self.friendIndex),\
                    group[2]) for group in candidateGroups]) < 5:

                lureDist = float('inf')
                action = Directions.STOP
                for a in getLimitedAction(gameState, self.index):
                    successor = gameState.generateSuccessor(self.index, a)
                    newDist = self.distancer.getDistance(successor.getAgentPosition(self.index), self.ghostStart)
                    if newDist < lureDist:
                        lureDist = newDist
                        action = a
                self.toBroadcast.append(action)
                return action
        ### Hard Code 2: born out ###
        if pacX <= self.bornHardLevel * 2 - 1 and pacX % 2 == 1:
            if (pacX + 1) / 2 % 2 == 1 and pacY != self.walls.height - 2:
                return Directions.NORTH
            elif (pacX + 1) / 2 % 2 == 0 and pacY != 1:
                return Directions.SOUTH
        ### Plan Agent ###
        if (self.followPlanFlag==True or self.forceFlag==True)\
                and self.distancer.getDistance(friendPos, (pacX, pacY)) >= 5:
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
            #print("reflex")
            self.plan=[]
            return self.gameTreeAction(gameState)
        self.toBroadcast = self.plan
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
        if self.DeadEndGrid[pacman[0]][pacman[1]]==True and ghostDist>=3:
            index=(-1,-1)
            for i in range(len(self.DeadEndRegionList)):
                for j in range(len(self.DeadEndRegionList[i])-1):
                    if pacman==self.DeadEndRegionList[i][j]:
                        index=(i,j+1)
            if index!=(-1,-1):
                return self.chooseNeighbourActioin(pacman,self.DeadEndRegionList[index[0]][index[1]])
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
        friendTargets = [(group, self.distancer.getDistance(friendPos, group[2])) \
                         for group in self.foodGroup if group[2] and len(group[0]) > 0 \
                         and self.distancer.getDistance(friendPos, group[2]) < 5]
        numTargets = sum([len(ft[0][0]) for ft in friendTargets])
        if numFood - numTargets > 3:
            for ft in friendTargets:
                for food in ft[0][0]:
                    if food in foods:
                        # self.debugDraw(food, [1,1,1])
                        foods.remove(food)
        # FIXME: too many removed?

        ghostTargets = [food for food in foods if self.distancer.getDistance(ghostPos, food) < 4]
        targetsNearTargets = []
        for f in ghostTargets:
            targetsNearTargets.extend([food for food in foods if self.distancer.getDistance(f, food) < 3])
        ghostTargets.extend(targetsNearTargets)
        ghostTargets = set(ghostTargets)
        if numFood - len(ghostTargets) > 3:
            for f in ghostTargets:
                if f in foods: foods.remove(f)

        #print("len of closed targets",len(self.closedTargets))
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

    ### Methods for gametree ###
    def gameTreeAction(self, gameState):
        if self.lastEvaluated:
            for a in self.lastGameState.getLegalActions(self.friendIndex):
                successor = self.lastGameState.generateSuccessor(self.friendIndex, a)
                if successor.getAgentPosition(self.friendIndex) == gameState.getAgentPosition(self.friendIndex):
                    self.friendBehavior = a
                    break
            if self.friendBehavior == self.friendPrediction[0]:
                self.simpleRightTimes += 1
            self.simpleEvalTimes += 1
            self.lastEvaluated = False

        self.terminal(gameState, self.index, 1, saveAction=True)
        if self.simpleEvalTimes < 10:
            self.lastEvaluated = True
            self.lastGameState = gameState
        elif self.simpleEvalTimes == 10:
            accuracy = float(self.simpleRightTimes / self.simpleEvalTimes)
            if accuracy < 0.3:
                self.friendIsStupid = False
        self.toBroadcast.append(self.decision)
        return self.decision

    def evaluation(self, gameState, ghostAction):
        ghostPos = gameState.getAgentPosition(self.ghostIndex)
        friendPos = gameState.getAgentPosition(self.friendIndex)
        myPos = gameState.getAgentPosition(self.index)
        if ghostPos != None and myPos != None:
            ghostToMe = self.distancer.getDistance(ghostPos, myPos)
        else:
            ghostToMe = -1
        if ghostPos != None and friendPos != None:
            ghostToFriend = self.distancer.getDistance(ghostPos, friendPos)
        else:
            ghostToFriend = -1
        if friendPos != None and myPos != None:
            friendToMe = self.distancer.getDistance(myPos, friendPos)
        else:
            friendToMe = -1
        foods = gameState.getFood().asList()

        ###Ghost Evaluation###
        ghostFeats = Counter()
        ghostWeight = {'successorScore': 10000, 'invaderDistance': -1, \
                       'stop': -100, 'reverse': 0, 'safeZone': -100000000}
        successorScore = -self.getScore(gameState)
        ghostFeats['successorScore'] = successorScore
        if ghostToMe != -1 and ghostToFriend != -1:
            ghostFeats['invaderDistance'] = min(ghostToFriend, ghostToMe)
        elif ghostToMe == -1 and ghostToFriend != -1:
            ghostFeats['invaderDistance'] = ghostToFriend
        elif ghostToMe != -1 and ghostToFriend == -1:
            ghostFeats['invaderDistance'] = ghostToMe
        if ghostAction == Directions.STOP: ghostFeats['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.ghostIndex).configuration.direction]
        if ghostAction == rev: ghostFeats['reverse'] = 1
        isSafeZone = max(12 - gameState.getAgentPosition(self.ghostIndex)[0], 0)
        ghostFeats['isSafeZone'] = isSafeZone
        ghostEval = ghostFeats * ghostWeight

        ###Friend Evaluation###
        if self.friendIsStupid:
            friendWeight = {'numFood': 1, 'closestFoodReward': 1}
            friendFeats = Counter()
            friendFeats['numFood'] = -len(foods)
            closestFood = min(self.distancer.getDistance(friendPos, food) for food in foods) + 2.0 \
                if len(foods) > 0 else 1.0
            friendFeats['closestFoodReward'] = 1.0 / closestFood
            friendEval = friendFeats * friendWeight
        else:
            friendWeight = {'numFood': 1, 'closestFoodReward': 1,
                            'closestGhost': -0.5, 'closestFriend': -1}
            friendFeats = Counter()
            friendFeats['numFood'] = -len(foods)
            # Remove dangerous food as the smarter staff bot
            currentDangerFood = []
            for f in foods:
                if f in self.dangerFood: currentDangerFood.append(f)
            closestFoodDist = min(10 * self.distancer.getDistance(friendPos, food) for food in currentDangerFood) \
                              + 2.0 if len(currentDangerFood) > 0 else 1.0
            closestFoodDist = min(closestFoodDist, min(self.distancer.getDistance(friendPos, food) \
                                                       for food in foods if food not in self.dangerFood) + 2.0 \
                if len(foods) != len(currentDangerFood) else 1.0)

            friendFeats['closestFoodReward'] = 1.0 / closestFoodDist
            friendFeats['closestGhost'] = 1.0 / (ghostToFriend ** 2) if ghostToMe < 20 else 0
            friendFeats['closestFriend'] = 1.0 / ((friendToMe + 0.01) ** 2) if friendToMe < 5 else 0
            friendEval = friendFeats * friendWeight

        ###My Evaluation###
        myWeight = {'numFood': 1, 'closestFood': 1, 'freeLevel': 0.1, 'foodDangerRegionReward': 0.1,
                    'closestGhost': -0.5, 'closestFriend': -0.5, 'deadEndPenalty': -1, 'subDeadEndPenalty': -0.5}
        #FIXME: tune the closestGhost according to ghost's distance?
        myFeats = Counter()
        myFeats['numFood'] = -len(foods)

        # Remove friend's target
        friendTargets = [(group, self.distancer.getDistance(friendPos, group[2]))\
                         for group in self.foodGroup if group[2] and len(group[0]) > 0\
                         and self.distancer.getDistance(friendPos, group[2]) < 5]
        numTargets = sum([len(ft[0][0]) for ft in friendTargets])
        if -myFeats['numFood'] - numTargets > 3:
            for ft in friendTargets:
                for food in ft[0][0]:
                    if food in foods:
                        #self.debugDraw(food, [1,1,1])
                        foods.remove(food)
        #FIXME: too many removed?

        # Remove really dangerous food
        if -myFeats['numFood'] > 10:
            foodWithEval = [(food, self.foodEval(gameState, food)) for food in foods]
            dangerousSet = set()
            for f in foodWithEval:
                if f[1] == -1:
                    dangerousSet.add(f)
            for f in dangerousSet:
                foodWithEval.remove(f)
                self.debugDraw(f[0], [1,0,0])
            if len(foodWithEval) == 0:
                myFeats['closestFood'] = 30
            else:
                closestFoodDist = self.distancer.getDistance(myPos, foodWithEval[0][0]) / foodWithEval[0][1]
                theFood = foods[0][0]
                for food, co in foodWithEval:
                    newDist = self.distancer.getDistance(myPos, food) / co
                    if newDist < closestFoodDist:
                        closestFoodDist = newDist
                        theFood = food
                closestFood = closestFoodDist + 2.0 if len(food) > 0 else 1.0
                myFeats['closestFood'] = 3 / closestFood
        else:
            closestFoodDist = min([self.distancer.getDistance(myPos, f) for f in foods])
            myFeats['closestFood'] = 3 / closestFoodDist
        myFeats['closestGhost'] = self.bornHardLevel*0.5 + 1 / (ghostToMe ** 2) if ghostToMe < 20 else 0
        myFeats['closestFriend'] = 1.0 / ((friendToMe + 0.01) ** 2) if friendToMe < 5 else 0

        if myFeats['closestGhost'] > 0.1:
            restriction = 0
            for offset in dir:
                if self.walls[myPos[0] + offset[0]][myPos[1] + offset[1]]:
                    restriction += 1
            myFeats['freeLevel'] = 1.0 / (restriction + 1.0)

        DeadEndGrid = 0
        # if self.DeadEndGrid[pacman[0]][pacman[1]]==True:
        if myPos in self.DeadEndRegion:
            myFeats['deadEndPenalty'] = 1

        subDeadEndGrid = 0
        if myPos in self.subDeadEndRegion:
            myFeats['subDeadEndPenalty'] = 1

        foodDangerRegionReward = 0
        pacLookUp = self.lookUpList(self.dangerFoodRegionList, myPos)
        if pacLookUp[0]:
            myFeats['foodDangerRegionReward'] = len(self.dangerFoodRegionList[pacLookUp[1][0]])

        myEval = myFeats * myWeight

        return myEval, friendEval, ghostEval

    def terminal(self, state, index, layer, action=None, saveAction=False):
        if layer == 0:
            return self.evaluation(state, action)
        else:
            if index == self.index:
                evalToGet = 0
            elif index == self.friendIndex:
                evalToGet = 1
                # if not self.friendIsStupid and self.receivedBroadcast != None and len(self.receivedBroadcast) > 2 - layer:
                if self.receivedBroadcast != None and len(self.receivedBroadcast) > 2 - layer:
                    ghost = state.getAgentPosition(self.ghostIndex)
                    friend = state.getAgentPosition(self.friendIndex)
                    actions = state.getLegalActions(self.friendIndex)
                    if self.distancer.getDistance(ghost, friend) >= 10 and self.receivedBroadcast[2 - layer] in actions:
                        nextState = state.generateSuccessor(self.friendIndex, self.receivedBroadcast[2 - layer])
                        return self.terminal(nextState, self.ghostIndex, layer)
            else:
                evalToGet = 2
                actions = getLimitedAction(state, self.ghostIndex)
                if random.random() < RANDOM_ACTION_PROB:
                    return self.expectedValue(state, layer)
            return self.maxValue(state, evalToGet, layer, saveAction)

    def expectedValue(self, gameState, layer):
        actions = getLimitedAction(gameState, self.ghostIndex)
        v0 = 0
        v1 = 0
        v2 = 0
        for a in actions:
            successor = gameState.generateSuccessor(self.ghostIndex, a)
            values = self.terminal(gameState, self.index, layer - 1, a)
            v0 += values[0]
            v1 += values[1]
            v2 += values[2]
        return v0 / len(actions), v1 / len(actions), v2 / len(actions)

    def maxValue(self, gameState, evalToGet, layer, saveAction):
        if evalToGet == 2:
            nextLayer = layer - 1
        else:
            nextLayer = layer
        v = float('-inf')
        ###Ghost Actions###
        if evalToGet == 2:
            if random.random() < RANDOM_ACTION_PROB:
                actions = getLimitedAction(gameState, self.ghostIndex)
            else:
                actions = gameState.getLegalActions(self.ghostIndex)
            maxValues = []
            for a in actions:
                successor = gameState.generateSuccessor(self.ghostIndex, a)
                value = self.terminal(successor, self.index, nextLayer, a)
                if v < value[evalToGet]:
                    maxValues = []
                    maxValues.append((value, a))
                    v = value[evalToGet]
                elif v == value[evalToGet]:
                    maxValues.append((value, a))
            rtn = random.choice(maxValues)
            if layer == 2:
                self.debugDraw(gameState.generateSuccessor(self.ghostIndex, rtn[1]).getAgentPosition(self.ghostIndex),
                               [0, 0, 1])
            if saveAction: self.decision = rtn[1]
            return rtn[0]

        ###Friend&My Actions###
        index = self.friendIndex if evalToGet == 1 else self.index
        nextIndex = self.ghostIndex if evalToGet == 1 else self.friendIndex
        actions = getLimitedAction(gameState, index)
        rtn = None
        decision = Directions.STOP
        for a in actions:
            successor = gameState.generateSuccessor(index, a)
            value = self.terminal(successor, nextIndex, nextLayer)
            if v < value[evalToGet]:
                v = value[evalToGet]
                decision = a
                rtn = value
        if layer == 2 and evalToGet == 1:
            self.debugDraw(gameState.generateSuccessor(self.friendIndex, decision).getAgentPosition(self.friendIndex),
                           [0, 1, 0])
            self.getNewPrediction(decision)
        if layer == 2 and evalToGet == 0:
            self.debugDraw(gameState.generateSuccessor(self.index, decision).getAgentPosition(self.index), [1, 0, 0])
        if saveAction: self.decision = decision
        return rtn



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

def getLimitedAction(state, index):
    """
    Actions without reverse and stop
    """
    legalActions = state.getLegalActions(index)
    legalActions.remove('Stop')
    if len(legalActions) > 1:
        rev = Directions.REVERSE[state.getAgentState(index).configuration.direction]
        if rev in legalActions:
            legalActions.remove(rev)
    return legalActions

def mannhattanDistance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
