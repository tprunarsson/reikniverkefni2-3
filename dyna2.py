#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dyna-2 using the function approximator architecture developed in part 1. Illustrate
the performance of this player against the random player during learning.
Compare the performance of this player against the neural network from part
1.

Googla hot one encoding 
"""
import Backgammon

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon


#def learn(board_0, Q, n):
#
#   #Initialise A,B -> Transition and reward models
#   
    #Clear permanent memory
#   theta = 0 
#
#   #loop   
#   for i in range(n) 
#       
#       #Start new episode
#       state = board_0
#
#       #Clear transient memory
#       theta_transient = 0
#
#       #Clear eligibility trace
#       z = 0
#
#       search(s)
#
#       action = pi(s,Q)
#
#       #while s is not terminal do
#           #Execute a, observe reward r, state s'
#           #next_state, reward = ?
#
#           #A(s,a) = next_state, B(s,a) = reward
#           (A,B) = updateModel(state,action,reward, next_state)  
#
#           
#           
#
#
#
#
#
#
#
#
#
#
#
#
#
#


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
#
#
    
    # policy missing, returns a random move for the time being
    #
    #
    #
    #
    #
    move = possible_moves[np.random.randint(len(possible_moves))]

    return move