#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import torch
import neural_network_agent
import Backgammon

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
    epsilon = 0.1
    #w1 = torch.load('./w1_trained.pth', map_location=lambda storage, loc: storage)
    #w2 = torch.load('./w2_trained.pth', map_location=lambda storage, loc: storage)
    #b1 = torch.load('./b1_trained.pth', map_location=lambda storage, loc: storage)
    #b2 = torch.load('./b2_trained.pth', map_location=lambda storage, loc: storage)
    
    w1 = torch.load('./w1_trained_first_time_working.pth', map_location=lambda storage, loc: storage)
    w2 = torch.load('./w2_trained_first_time_working.pth', map_location=lambda storage, loc: storage)
    b1 = torch.load('./b1_trained_first_time_working.pth', map_location=lambda storage, loc: storage)
    b2 = torch.load('./b2_trained_first_time_working.pth', map_location=lambda storage, loc: storage)
    move = neural_network_agent.epsilon_nn_greedy(board_copy, dice, player, epsilon, w1, b1, w2, b2, possible_moves, possible_boards, False)

    return move