# multiAgents.py
# --------------
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


from typing import Tuple
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

# import mean function
from statistics import mean

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
                
        eval = 0
        
        ## First, check if the new state is a ghost state, if so, return -inf
        ## Second, check the distance to the closest food, deduce it form the eval
        
        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) <= 1:
                return -float('inf')
        
        foodList = newFood.asList()
        if len(foodList) > 0:
            closestFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            eval -= closestFoodDist
        
        has_less_food = currentGameState.getNumFood() > successorGameState.getNumFood()
        
        # Check if there is a power pellet nearby
        powerPellets = currentGameState.getCapsules()
        if len(powerPellets) > 0:
            closestPelletDist = min([manhattanDistance(newPos, pellet) for pellet in powerPellets])
            eval -= closestPelletDist * 1000 / 2  # Less weight for pellets
            
        # If the powerpellet is active, chase the ghosts
        for i, ghost in enumerate(newGhostStates):
            if newScaredTimes[i] > 0:  # Ghost is scared
                dist = manhattanDistance(newPos, ghost.getPosition())
                if dist > 0:
                    eval += 500 / dist  # More weight for chasing scared ghosts
        

        return eval + successorGameState.getScore() + has_less_food * 1000
        
        
        
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        
        depth = self.depth
        eval = self.evaluationFunction
        num_agents = gameState.getNumAgents()
        
        final_action = None

        def get_best_move(agent_index, state: GameState, curr_depth) -> Tuple[float, Directions]:
            if state.isWin() or state.isLose() or curr_depth == depth:
                return eval(state), None

            actions = state.getLegalActions(agent_index)
            if len(actions) == 0:
                return eval(state), None

            next_index = (agent_index + 1) % num_agents

            next_depth = curr_depth + 1 if next_index == 0 else curr_depth

            best_action = None

            if agent_index == 0:  # MAX node (Pacman)
                best_score = -float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agent_index, action)
                    score, _ = get_best_move(next_index, successor, next_depth)
                    if score > best_score:
                        best_score = score
                        best_action = action
            else:  # MIN node (ghost)
                best_score = float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agent_index, action)
                    score, _ = get_best_move(next_index, successor, next_depth)
                    if score < best_score:
                        best_score = score
                        best_action = action

            return best_score, best_action

        _, final_action = get_best_move(0, gameState, 0)
        return final_action
        
        
        
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        depth = self.depth
        eval = self.evaluationFunction
        num_agents = gameState.getNumAgents()
        
        final_action = None

        def get_best_move(agent_index, state: GameState, curr_depth, alpha: float, beta: float) -> Tuple[float, Directions]:
            if state.isWin() or state.isLose() or curr_depth == depth:
                return eval(state), None

            actions = state.getLegalActions(agent_index)
            if len(actions) == 0:
                return eval(state), None

            next_index = (agent_index + 1) % num_agents

            next_depth = curr_depth + 1 if next_index == 0 else curr_depth

            best_action = None

            if agent_index == 0:  # MAX node (Pacman)
                best_score = -float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agent_index, action)
                    score, _ = get_best_move(next_index, successor, next_depth, alpha, beta)
                    if score > best_score:
                        best_score = score
                        best_action = action
                    alpha = max(alpha, best_score)
                    if beta < alpha:
                        break 
            else:  # MIN node (ghost)
                best_score = float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agent_index, action)
                    score, _ = get_best_move(next_index, successor, next_depth, alpha, beta)
                    if score < best_score:
                        best_score = score
                        best_action = action
                    beta = min(beta, best_score)
                    if beta < alpha:
                        break

            return best_score, best_action

        _, final_action = get_best_move(0, gameState, 0, -float('inf'), float('inf'))
        return final_action
        
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        
        depth = self.depth
        eval = self.evaluationFunction
        num_agents = gameState.getNumAgents()
        
        final_action = None

        def get_best_move(agent_index, state: GameState, curr_depth) -> Tuple[float, Directions]:
            if state.isWin() or state.isLose() or curr_depth == depth:
                return eval(state), None

            actions = state.getLegalActions(agent_index)
            if len(actions) == 0:
                return eval(state), None

            next_index = (agent_index + 1) % num_agents

            next_depth = curr_depth + 1 if next_index == 0 else curr_depth

            best_action = None

            if agent_index == 0:  # MAX node (Pacman)
                best_score = -float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agent_index, action)
                    score, _ = get_best_move(next_index, successor, next_depth)
                    if score > best_score:
                        best_score = score
                        best_action = action
            else:  # MIN node (ghost)
                best_score = float('inf')
                scores = []
                for action in actions:
                    successor = state.generateSuccessor(agent_index, action)
                    score, _ = get_best_move(next_index, successor, next_depth)
                    scores.append(score)
                best_score = mean(scores)

            return best_score, best_action

        _, final_action = get_best_move(0, gameState, 0)
        return final_action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Considering the same params as on first task, but based on the state, not on the possible actions..
    """
    "*** YOUR CODE HERE ***"
    
    
    pacmanPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    
    ev = score

    foodList = currentGameState.getFood().asList()
    if foodList:
        foodDistances = [util.manhattanDistance(pacmanPos, food) for food in foodList]
        minFoodDistance = min(foodDistances)
        ev += 50.0 / (minFoodDistance + 1) # Add 1 to avoid division by zero

    ev -= 100 * currentGameState.getNumFood()

    killswitches = currentGameState.getCapsules()
    if killswitches:
        killswitch_distances = [util.manhattanDistance(pacmanPos, cap) for cap in killswitches]
        min_killswitch_distance = min(killswitch_distances)

        ev += 20.0 / (min_killswitch_distance + 1)

        ev -= 20 * len(killswitches)
        
    ghostStates = currentGameState.getGhostStates()
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ((distance_to_goshts)) = util.manhattanDistance(pacmanPos, ghostPos)
        timeRemaining = ghostState.scaredTimer
        
        if ((distance_to_goshts)) == 0:
            # Houston, we have a problem (either eaten or eating)
            pass

        elif timeRemaining > 0:
            ev += 4.0 / (((distance_to_goshts)) + 1)
        
        else:
            if ((distance_to_goshts)) <= 2:
                 ev -= 2000.0 / ((((distance_to_goshts))) ** 2)
            else:
                 ev -= 50.0 / (((distance_to_goshts)) + 1)

    return ev

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
