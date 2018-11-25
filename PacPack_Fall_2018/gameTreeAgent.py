"""
An agent using non-zerosum game tree strategy.
Currently hardcode for friend and ghost.
"""

from util import Counter
from game import Actions, Directions
import random
from captureAgents import CaptureAgent

RANDOM_ACTION_PROB = 0.4
dir = []
for i in reversed(range(-1, 2)):
    for j in range(-1, 2):
        dir.append((j, i))

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
        self.walls = gameState.getWalls()
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


    def evaluation(self, gameState, ghostAction):
        ghostPos = gameState.getAgentPosition(self.ghostIndex)
        friendPos = gameState.getAgentPosition(self.friendIndex)
        myPos = gameState.getAgentPosition(self.index)
        if ghostPos != None and myPos != None:
            ghostToMe = self.distancer.getDistance(ghostPos, myPos)
        else: ghostToMe = -1
        if ghostPos != None and friendPos != None:
            ghostToFriend = self.distancer.getDistance(ghostPos, friendPos)
        else: ghostToFriend = -1
        if friendPos != None and myPos != None:
            friendToMe = self.distancer.getDistance(myPos, friendPos)
        else: friendToMe = -1
        foods = gameState.getFood().asList()

        ###Ghost Evaluation###
        ghostFeats = Counter()
        ghostWeight = {'successorScore': 10000, 'invaderDistance': -1,\
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
        friendWeight = {'numFood': 1, 'closestFoodReward': 1}
        friendFeats = Counter()
        friendFeats['numFood'] = -len(foods)
        closestFood = min(self.distancer.getDistance(friendPos, food) for food in foods) + 2.0 \
            if len(foods) > 0 else 1.0
        friendFeats['closestFoodReward'] = 1.0 / closestFood
        friendEval = friendFeats * friendWeight

        ###My Evaluation###
        myWeight = {'numFood': 1, 'closestFood': 1,
                   'closestGhost': -1, 'closestFriend': -1}
        myFeats = Counter()
        myFeats['numFood'] = -len(foods)

        closestFoodDist = self.distancer.getDistance(myPos, foods[0])
        theFood = foods[0]
        for food in foods:
            newDist = self.distancer.getDistance(myPos, food)
            if newDist < closestFoodDist:
                closestFoodDist = newDist
                theFood = food
        coefficient = self.foodEval(gameState, food, ghostPos, friendPos)
        closestFood = closestFoodDist + 2.0 if len(foods) > 0 else 1.0
        myFeats['closestFood'] = 3 * coefficient / closestFood
        if myFeats['closestFood'] == 0: myFeats['numFood'] = -60

        # availableActions = getLimitedAction(gameState, self.index)
        # freeLevel = 0
        # for a in availableActions:
        #     successor = gameState.generateSuccessor(self.index, a)
        #     nextActions = len(getLimitedAction(successor, self.index))
        #     freeLevel += nextActions + 1
        # myFeats['freeLevel'] = freeLevel
        myFeats['closestGhost'] = 1.0 / (ghostToMe ** 2) if ghostToMe < 20 else 0
        myFeats['closestFriend'] = 1.0 / ((friendToMe+0.01) ** 2) if friendToMe < 5 else 0

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
            else:
                evalToGet = 2
            return self.maxValue(state, evalToGet, layer, saveAction)

    def maxValue(self, gameState, evalToGet, layer, saveAction):
        if evalToGet == 2:
            nextLayer = layer - 1
        else: nextLayer = layer
        v = float('-inf')
        ###Ghost Actions###
        if evalToGet == 2:
            if random.random() <RANDOM_ACTION_PROB:
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
                self.debugDraw(gameState.generateSuccessor(self.ghostIndex, rtn[1]).getAgentPosition(self.ghostIndex), [0,0,1])
                print(gameState.generateSuccessor(self.ghostIndex, rtn[1]).getAgentPosition(self.ghostIndex))
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
            self.debugDraw(gameState.generateSuccessor(self.friendIndex, decision).getAgentPosition(self.friendIndex), [0,1,0])
        if layer == 2 and evalToGet == 0:
            self.debugDraw(gameState.generateSuccessor(self.index, decision).getAgentPosition(self.index), [1,0,0])
        if saveAction: self.decision = decision
        return rtn

    def chooseAction(self, gameState):
        self.debugClear()
        pacX, pacY = gameState.getAgentPosition(self.index)
        if pacX <= self.bornHardLevel * 2 - 1 and pacX % 2 == 1:
            if (pacX+1) / 2 % 2 == 1 and pacY != self.walls.height - 2:
                return Directions.NORTH
            elif (pacX+1) / 2 % 2 == 0 and pacY != 1:
                return Directions.SOUTH
        self.terminal(gameState, self.index, 2, saveAction=True)
        return self.decision

    ###Helper Functions###

    def foodEval(self, gameState, food, ghost, friend):
        #FIXME: How to consider my friend
        #FIXME: How to evaluate the coefficient correctly
        walls = gameState.getWalls()
        foods = gameState.getFood()
        foodX, foodY = food
        ghostAround = False
        friendAround = False
        cornerAround = 0
        foodAround = 0
        for offset in dir:
            dx, dy = offset
            if walls[dx + foodX][dy + foodY]:
                cornerAround += 1
            if foods[dx + foodX][dy + foodY]:
                foodAround += 1
        if cornerAround < 6: cornerAround = 0
        ghostDist = self.distancer.getDistance(food, ghost)
        if ghostDist < 4:
            ghostAround = True
        friendDist = self.distancer.getDistance(food, friend)
        if friendDist < 4:
            friendAround = True
        if ghostAround and cornerAround != 0:
            if cornerAround == 6: return 0.6 + foodAround * 0.05
            elif cornerAround == 7: return 0.3 + friendAround * 0.05
            elif cornerAround > 7: return 0
        elif cornerAround != 0:
            if cornerAround == 6: return 0.9 + foodAround * 0.1
            elif cornerAround == 7: return 0.6 + friendAround * 0.1
            elif cornerAround > 7: return 0.5 + foodAround * 0.05
        return 1






def getLimitedAction(state, index):
    legalActions = state.getLegalActions(index)
    legalActions.remove('Stop')
    if len(legalActions) > 1:
        rev = Directions.REVERSE[state.getAgentState(index).configuration.direction]
        if rev in legalActions:
            legalActions.remove(rev)
    return legalActions
