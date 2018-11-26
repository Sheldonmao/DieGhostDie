"""
An agent using non-zerosum game tree strategy.
Currently hardcode for friend and ghost.
@Author: Zhibo Fan
"""

from util import Counter
from game import Actions, Directions
import random
from captureAgents import CaptureAgent

RANDOM_ACTION_PROB = 0.4
dir = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GameTreeAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.actionToTake = Directions.NORTH
        self.ghostIndex = gameState.getGhostTeamIndices()[0]
        self.friendIndex = -1
        for i in gameState.getPacmanTeamIndices():
            if i != self.index:
                self.friendIndex = i
                break
        self.ghostStart = gameState.getAgentPosition(self.ghostIndex)
        self.walls = gameState.getWalls()
        self.foodGrid = gameState.getFood()
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
        self.dangerFood = list()

        for food in self.foodGrid.asList():
            wallsAround = 0
            for offset in dir:
                x = food[0] + offset[0]
                y = food[1] + offset[1]
                if self.walls[x][y]:
                    wallsAround += 1
            if wallsAround >= 3: self.dangerFood.append(food)
        self.dangerFood = set(self.dangerFood)
        self.dangerRegion = []
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
            self.foodGroup.append((thisGroup, regionSize + 1, tempPos))
        problematicFood = set()
        for i in range(1, len(self.foodGroup)):
            for f in self.foodGroup[i][0]: problematicFood.add(f)
        safeGroup = set([food for food in gameState.getFood().asList() if food not in problematicFood])
        self.foodGroup[0] = (safeGroup, 0, None)

        ###Atributes for which evaluation to use for friend###
        self.friendIsStupid = False
        self.simpleEvalTimes = 0
        self.simpleRightTimes = 0
        self.lastGameState = None
        self.friendBehavior = Directions.STOP
        self.friendPrediction = [Directions.STOP, Directions.STOP]
        self.lastEvaluated = False
        self.toBroadcast = []

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
        myWeight = {'numFood': 1, 'closestFood': 1, 'freeLevel': 0.1,
                    'closestGhost': -0.5, 'closestFriend': -0.5}
        myFeats = Counter()
        myFeats['numFood'] = -len(foods)

        # Remove friend's target
        friendTargets = [food for food in foods if self.distancer.getDistance(friendPos, food) < 7]
        targetsNearTargets = []
        for f in friendTargets:
            targetsNearTargets.extend([food for food in foods if self.distancer.getDistance(f, food) < 4])
        friendTargets.extend(targetsNearTargets)
        friendTargets = set(friendTargets)
        if -myFeats['numFood'] - len(friendTargets) > 3:
            for f in friendTargets:
                if f in foods: foods.remove(f)

        # Remove really dangerous food
        if -myFeats['numFood'] > 10:
            for f in self.dangerFood:
                if f in foods and self.distancer.getDistance(ghostPos, f) < 8:
                    foods.remove(f)
        if len(foods) == 0:
            myFeats['closestFood'] = 30
        else:
            closestFoodDist = self.distancer.getDistance(myPos, foods[0])
            theFood = foods[0]
            for food in foods:
                newDist = self.distancer.getDistance(myPos, food)
                if newDist < closestFoodDist:
                    closestFoodDist = newDist
                    theFood = food
            coefficient = self.foodEval(gameState, food, ghostPos, friendPos)
            closestFood = closestFoodDist + 2.0 if len(food) > 0 else 1.0
            myFeats['closestFood'] = 3 * coefficient / closestFood
        if myFeats['closestFood'] == 0: myFeats['numFood'] = -60
        myFeats['closestGhost'] = 1.0 / (ghostToMe ** 2) if ghostToMe < 20 else 0
        myFeats['closestFriend'] = 1.0 / ((friendToMe + 0.01) ** 2) if friendToMe < 5 else 0

        if myFeats['closestGhost'] > 0.1:
            restriction = 0
            for offset in dir:
                if self.walls[myPos[0] + offset[0]][myPos[1] + offset[1]]:
                    restriction += 1
            myFeats['freeLevel'] = 1.0 / (restriction + 1.0)

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
                    action = random.choice(actions)
                    nextState = state.generateSuccessor(self.ghostIndex, action)
                    return self.terminal(nextState, self.index, layer-1, action)
            return self.maxValue(state, evalToGet, layer, saveAction)

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

    def chooseAction(self, gameState):
        self.debugClear()
        self.toBroadcast = []
        pacX, pacY = gameState.getAgentPosition(self.index)
        ghostPos = gameState.getAgentPosition(self.ghostIndex)
        if pacX >= 32 - self.bornHardLevel * 2 \
                and self.distancer.getDistance((pacX, pacY), ghostPos) < 7:
            currentClusters = [f for f in self.clusteredFood if f in gameState.getFood().asList()]
            if self.distancer.getDistance((pacX, pacY), self.ghostStart) \
                    < self.distancer.getDistance(ghostPos, self.ghostStart) \
                    and len(currentClusters) > 0 \
                    and min([self.distancer.getDistance(gameState.getAgentPosition(self.friendIndex), \
                                                        f) for f in currentClusters]) < 5:

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
        if pacX <= self.bornHardLevel * 2 - 1 and pacX % 2 == 1:
            if (pacX + 1) / 2 % 2 == 1 and pacY != self.walls.height - 2:
                return Directions.NORTH
            elif (pacX + 1) / 2 % 2 == 0 and pacY != 1:
                return Directions.SOUTH

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

        self.terminal(gameState, self.index, 2, saveAction=True)
        if self.simpleEvalTimes < 10:
            self.lastEvaluated = True
            self.lastGameState = gameState
        elif self.simpleEvalTimes == 10:
            accuracy = float(self.simpleRightTimes / self.simpleEvalTimes)
            if accuracy < 0.3:
                self.friendIsStupid = False
        self.toBroadcast.append(self.decision)
        return self.decision

    ###Helper Functions###

    def foodEval(self, gameState, food):
        for group in range(len(self.foodGroup)):
            if food in self.foodGroup[group][0]:
                return self.foodEvalGroup(gameState, group)

    def foodEvalGroup(self, gameState, group):
        ########################################################################
        #    If food is too dangerous to eat (need to remove),                 #
        #    return -1,                                                        #
        #    else consider the distance(exit, ghost) and region_size / #food.  #
        ########################################################################
        if group == 0: return 1
        foodSet, regionSize, exit = self.foodGroup[group]
        ghostDist = self.distancer.getDistance(gameState.getAgentPosition(
            gameState.getGhostTeamIndices()[0]), exit)
        if ghostDist < 1.5 * regionSize: return -1
        ghostFactor = min(ghostDist, 3 * regionSize, 20) / min(3 * regionSize, 20)
        smooth = 18 #FIXME: tune it
        worth = (len(foodSet) + smooth) / (regionSize * 2 + smooth)
        return ghostFactor * smooth

    def getNewPrediction(self, a):
        self.friendPrediction[0] = self.friendPrediction[1]
        self.friendPrediction[1] = a


def mannhattanDistance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def getLimitedAction(state, index):
    legalActions = state.getLegalActions(index)
    legalActions.remove('Stop')
    if len(legalActions) > 1:
        rev = Directions.REVERSE[state.getAgentState(index).configuration.direction]
        if rev in legalActions:
            legalActions.remove(rev)
    return legalActions
