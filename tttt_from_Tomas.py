import numpy as np
import torch 
from torch.autograd import Variable

# this function just works out the winner of the game
def iswin(board, m):
    if np.all(board[[0, 1, 2]] == m) | np.all(board[[3, 4, 5]] == m):
        return 1
    if np.all(board[[6, 7, 8]] == m) | np.all(board[[0, 3, 6]] == m):
        return 1
    if np.all(board[[1, 4, 7]] == m) | np.all(board[[2, 5, 8]] == m):
        return 1
    if np.all(board[[0, 4, 8]] == m) | np.all(board[[2, 4, 6]] == m):
        return 1
    return 0

# this function finds all legal actions (moves) for given state A(s)
def legal_moves(board):
    return np.where(board == 0)[0]

# this function gets the other player in a turn taking game
def getotherplayer(player):
    if (player == 1):
        return 2
    return 1

# this function is used to find an index to the after-state value table V(s)
def hashit(board):
    base3 = np.matmul(np.power(3, range(0, 9)), board.transpose())
    return int(base3)

# the usual epsilon greedy policy
def epsilongreedy(board, player, epsilon, V, debug = False):
    moves = legal_moves(board)
    if (np.random.uniform() < epsilon):
        if debug == True:
            print("explorative move")
        return np.random.choice(moves, 1)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(0, na):
        board[moves[i]] = player
        va[i] = V[hashit(board)]
        board[moves[i]] = 0  # undo move
    return moves[np.argmax(va)]

# this function is used to prepare the raw board as input to the network
# for some games (not here) it may be useful to invert the board and see it from the perspective of "player"
def one_hot_encoding(board, player):
    one_hot = np.zeros( 2 * len(board) )
    one_hot[np.where(board == 1)[0] ] = 1
    one_hot[len(board) + np.where(board == 2)[0] ] = 1
    return one_hot

# this epsilon greedy policy uses a feed-forward neural network to approximate the after-state value function
def epsilon_nn_greedy(board, player, epsilon, w1, b1, w2, b2, debug = False):
    moves = legal_moves(board)
    if (np.random.uniform() < epsilon):
        if debug == True:
            print("explorative move")
        return np.random.choice(moves, 1)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(0, na):
        board[moves[i]] = player
        # encode the board to create the input
        x = Variable(torch.tensor(one_hot_encoding(board, player), dtype = torch.float, device = device)).view(2*9,1)
        # now do a forward pass to evaluate the board's after-state value
        h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        va[i] = y.sigmoid()
        board[moves[i]] = 0
    return moves[np.argmax(va)]

def learnit(numgames, epsilon, lam, alpha, V, alpha1, alpha2, w1, b1, w2, b2):
    gamma = 1 # for completeness
    # play numgames games for training
    for games in range(0, numgames):
        board = np.zeros(9)    # initialize the board (empty)
        # we will use TD(lambda) and so we need to use eligibility traces
        S = [] # no after-state for table V, visited after-states is an empty list
        E = np.array([]) # eligibility traces for table V
        # now we initilize all the eligibility traces for the neural network
        Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
        Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
        Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
        Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)
        # player to start is "1" the other player is "2"
        player = 1
        tableplayer = 2
        winner = 0 # this implies a draw
        # start turn playing game, maximum 9 moves
        for move in range(0, 9):
            # use a policy to find action
            if (player == tableplayer): # this one is using the table V
                action = epsilongreedy(np.copy(board), player, epsilon, V)
            else: # this one is using the neural-network to approximate the after-state value
                action = epsilon_nn_greedy(np.copy(board), player, epsilon, w1, b1, w2, b2)
            # perform move and update board
            board[action] = player
            if (1 == iswin(board, player)): # has this player won?
                winner = player
                break # bail out of inner game loop
            # once both player have performed at least one move we can start doing updates
            if (1 < move):
                if tableplayer == player: # here we have player 1 updating the table V
                    s = hashit(board) # get index to table for this new board
                    delta = 0 + gamma * V[s] - V[sold]
                    E = np.append(E,1) # add trace to this state (note all new states are unique else we would +1)
                    S.append(sold)     # keep track of this state also
                    print('S',S)
                    print('V[S]',V[S])
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
                sold = hashit(board)
            else:
                xold = Variable(torch.tensor(one_hot_encoding(board, player), dtype=torch.float, device = device)).view(2*9,1)
            # swap players
            player = getotherplayer(player)

        # The game epsiode has ended and we know the outcome of the game, and can find the terminal rewards
        if winner == tableplayer:
            reward = 0
        elif winner == getotherplayer(tableplayer):
            reward = 1
        else:
            reward = 0.5
        # Now we perform the final update (terminal after-state value is zero)
        # these are basically the same updates as in the inner loop but for the final-after-states (sold and xold)
        # first for the table (note if reward is 0 this player actually won!):
        delta = (1.0 - reward) + gamma * 0 - V[sold]
        E = np.append(E,1) # add one to the trace (recall unique states)
        S.append(sold)
        V[S] = V[S] + delta * alpha * E
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

# global after-state value function, note this table is too big and contrains states that
# will never be used, also each state is unique to the player (no one after-state seen by both players)
V = np.zeros(hashit(2 * np.ones(9)))

alpha = 0.01 # step size for tabular learning
alpha1 = 0.01 # step sizes using for the neural network (first layer)
alpha2 = 0.01 # (second layer)
epsilon = 0.1 # exploration parameter used by both players
lam = 0.4 # lambda parameter in TD(lam-bda)

# define the parameters for the single hidden layer feed forward neural network
# randomly initialized weights with zeros for the biases
w1 = Variable(torch.randn(9*9,2*9, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((9*9,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(torch.randn(1,9*9, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

# now perform the actual training and display the computation time
import time
start = time.time()
training_steps = 40000
learnit(training_steps, epsilon, lam, alpha, V, alpha1, alpha2, w1, b2, w2, b2)
end = time.time()
print(end - start)

def competition(V, w1, b1, w2, b2, epsilon = 0.0, debug = False):
    board = np.zeros(9)          # initialize the board
    # player to start is "1" the other player is "2"
    player = 1
    tableplayer = 2
    winner = 0 # default draw
    # start turn playing game, maximum 9 moves
    for move in range(0, 9):
        # use a policy to find action, switch off exploration
        if (tableplayer == player):
            action = epsilongreedy(np.copy(board), player, epsilon, V, debug)
        else:
            action = epsilon_nn_greedy(np.copy(board), player, epsilon, w1, b1, w2, b2, debug)
        # perform move and update board (for other player)
        board[action] = player
        if debug: # print the board, when in debug mode
            symbols = np.array([" ", "X", "O"])
            print("player ", symbols[player], ", move number ", move+1, ":", action)
            print(symbols[board.astype(int)].reshape(3,3))

        if (1 == iswin(board, player)): # has this player won?
            winner = player
            break
        player = getotherplayer(player) # swap players
    return winner
        
wins_for_player_1 = 0
draw_for_players = 0
loss_for_player_1 = 0
competition_games = 100
for j in range(competition_games):
    winner = competition(V, w1, b1, w2, b2, epsilon, debug = False)
    if (winner == 1):
        wins_for_player_1 += 1.0
    elif (winner == 0):
        draw_for_players += 1.0
    else:
        loss_for_player_1 += 1.0

print(wins_for_player_1, draw_for_players, loss_for_player_1)
# lets also play one deterministic game:
winner = competition(V, w1, b1, w2, b2, 0, debug = True)
