#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
from collections import defaultdict
import torch 
from torch.autograd import Variable
import Backgammon


# this function is used to find an index to the after-state value table V(s)
def hash_it(board_copy):
    base31 = np.matmul(np.power(31, range(0, 29)), board_copy.transpose())
    return int(base31)

def one_hot_encoding(board):
    oneHot = np.zeros(28*6*2)
    for i in range(1, 7):
        if i < 6:
            oneHot[28 * (i-1) + (np.where( board == i)[0] )-1] = 1
            oneHot[28*6 + 28 * (i-1) + (np.where( board == -i)[0] )-1] = 1
        else:
            oneHot[28 * (i-1) + (np.where( board > i)[0] )-1] = 1
            oneHot[28*6 + 28 * (i-1) + (np.where( board < -i)[0] )-1] = 1
    return oneHot


# this epsilon greedy policy uses a feed-forward neural network to approximate the after-state value function
def epsilon_nn_greedy(board, dice, player, epsilon, w1, b1, w2, b2, debug = False):
    possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
    if (np.random.uniform() < epsilon):
        if debug == True:
            print("explorative move")
        return possible_moves[np.random.randint(len(possible_moves))]
    na = len(possible_boards)
    va = np.zeros(na)
    for i in range(0, na):
        # encode the board to create the input
        x = Variable(torch.tensor(one_hot_encoding(possible_moves[i]), dtype = torch.float, device = device)).view(28*2*6,1)
        # now do a forward pass to evaluate the board's after-state value
        h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        va[i] = y.sigmoid()
    return possible_moves[np.argmax(va)]

def learnit(numgames, epsilon, lam, alpha, V, alpha1, alpha2, w1, b1, w2, b2):
    gamma = 1 # for completeness
    # play numgames games for training
    for games in range(0, numgames):
        board = Backgammon.init_board()    # initialize the board (empty)
        # we will use TD(lambda) and so we need to use eligibility traces
        S = [] # no after-state for table V, visited after-states is an empty list
        E = np.array([]) # eligibility traces for table V
        # now we initilize all the eligibility traces for the neural network
        Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
        Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
        Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
        Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)
        # player to start is "1" the other player is "-1"
        player = 1
        tableplayer = -1
        winner = 0 # this implies a draw
        # start turn playing game, maximum 9 moves
        dice = Backgammon.roll_dice()
        legal_moves = Backgammon.legal_moves(board, dice, player)
        for moveNumber in range(0, len(legal_moves)):
            # use a policy to find action
            if (player == tableplayer): # this one is using the table V
                possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
                action = possible_moves[np.random.randint(len(possible_moves))]
            else: # this one is using the neural-network to approximate the after-state value
                action = epsilon_nn_greedy(np.copy(board), dice, player, epsilon, w1, b1, w2, b2)
            # perform move and update board
            for i in range(0,len(action)):
                board = Backgammon.update_board(board, action[i], player)
            if (1 == Backgammon.game_over(board)): # has this player won?
                winner = player
                break # bail out of inner game loop
            # once both player have performed at least one move we can start doing updates
            if (1 < moveNumber):
                if tableplayer == player: # here we have player 1 updating the table V
                    s = hash_it(board) # get index to table for this new board
                    delta = 0 + gamma * V[s] - V[sold]
                    E = np.append(E,1) # add trace to this state (note all new states are unique else we would +1)
                    S.append(sold)     # keep track of this state also
                    V[S] = V[S] + delta * alpha * E # the usual tabular TD(lambda) update
                    E = gamma * lam * E
                else: # here we have player 2 updating the neural-network (2 layer feed forward with Sigmoid units)
                    x = Variable(torch.tensor(one_hot_encoding(board, player), dtype = torch.float, device = device)).view(2*9,1)
                    # now do a forward pass to evaluate the new board's after-state value
                    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
                    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
                    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
                    y_sigmoid = y.sigmoid() # squash this with a sigmoid function
                    target = y_sigmoid.detach().cpu().numpy()
                    # lets also do a forward past for the old board, this is the state we will update
                    h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
                    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
                    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
                    y_sigmoid = y.sigmoid() # squash the output
                    delta2 = 0 + gamma * target - y_sigmoid.detach().cpu().numpy() # this is the usual TD error
                    # using autograd and the contructed computational graph in pytorch compute all gradients
                    y_sigmoid.backward()
                    # update the eligibility traces using the gradients
                    Z_w2 = gamma * lam * Z_w2 + w2.grad.data
                    Z_b2 = gamma * lam * Z_b2 + b2.grad.data
                    Z_w1 = gamma * lam * Z_w1 + w1.grad.data
                    Z_b1 = gamma * lam * Z_b1 + b1.grad.data
                    # zero the gradients
                    w2.grad.data.zero_()
                    b2.grad.data.zero_()
                    w1.grad.data.zero_()
                    b1.grad.data.zero_()
                    # perform now the update for the weights
                    delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
                    w1.data = w1.data + alpha1 * delta2 * Z_w1
                    b1.data = b1.data + alpha1 * delta2 * Z_b1
                    w2.data = w2.data + alpha2 * delta2 * Z_w2
                    b2.data = b2.data + alpha2 * delta2 * Z_b2

            # we need to keep track of the last board state visited by the players
            if tableplayer == player:
                sold = hash_it(board)
            else:
                xold = Variable(torch.tensor(one_hot_encoding(board), dtype=torch.float, device = device)).view(28*2*6,1)
            # swap players
            player = -player

        # The game epsiode has ended and we know the outcome of the game, and can find the terminal rewards
        if winner == tableplayer:
            reward = 0
        elif winner == -tableplayer:
            reward = 1
        else:
            reward = 0.5
        # Now we perform the final update (terminal after-state value is zero)
        # these are basically the same updates as in the inner loop but for the final-after-states (sold and xold)
        # first for the table (note if reward is 0 this player actually won!):
        delta = (1.0 - reward) + gamma * 0 - V[sold]
        E = np.append(E,1) # add one to the trace (recall unique states)
        S.append(sold)

        for state in S:
            #          print('V[state]',V[state])
            V[state] = V[state] + delta * alpha * E
        # and then for the neural network:
        h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid() # squash the output
        delta2 = reward + gamma * 0 - y_sigmoid.detach().cpu().numpy()  # this is the usual TD error
        # using autograd and the contructed computational graph in pytorch compute all gradients
        y_sigmoid.backward()
        # update the eligibility traces
        Z_w2 = gamma * lam * Z_w2 + w2.grad.data
        Z_b2 = gamma * lam * Z_b2 + b2.grad.data
        Z_w1 = gamma * lam * Z_w1 + w1.grad.data
        Z_b1 = gamma * lam * Z_b1 + b1.grad.data
        # zero the gradients
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        w1.grad.data.zero_()
        b1.grad.data.zero_()
        # perform now the update of weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
        w1.data = w1.data + alpha1 * delta2 * Z_w1
        b1.data = b1.data + alpha1 * delta2 * Z_b1
        w2.data = w2.data + alpha2 * delta2 * Z_w2
        b2.data = b2.data + alpha2 * delta2 * Z_b2
        
device = torch.device('cpu')
# cuda will only create a significant speedup for large/deep networks and batched training
# device = torch.device('cuda') 

V = defaultdict(int) 

alpha = 0.01 # step size for tabular learning
alpha1 = 0.01 # step sizes using for the neural network (first layer)
alpha2 = 0.01 # (second layer)
epsilon = 0.1 # exploration parameter used by both players
lam = 0.4 # lambda parameter in TD(lam-bda)

# define the parameters for the single hidden layer feed forward neural network
# randomly initialized weights with zeros for the biases
w1 = Variable(torch.randn(28*28,28*2*6, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((28*28,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(torch.randn(1,28*28, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

# now perform the actual training and display the computation time
import time
start = time.time()
training_steps = 10000
learnit(training_steps, epsilon, lam, alpha, V, alpha1, alpha2, w1, b2, w2, b2)
end = time.time()
print(end - start)

def action(board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 
    
    # make the best move according to the policy
    
    # policy missing, returns a random move for the time being
    #
    #
    #
    #
    #
    move = epsilon_nn_greedy(np.copy(board), player, epsilon, w1, b1, w2, b2, debug)

    return move