# Backgammon

Backgammon interface for the 2nd and 3rd computational assignments in Computational Intelligence.

# Reikniverkefni 2-3

## Fyrir hópmeðlimi

### Neural Network Agent

(temp lausn munum gera þetta meira nice)

1. Commenta út import agent í Backgammon.py

```py
 # import agent.py
```

2.  Þjálfa neural_network_agent með því að keyra neural_network_agent.py þar sem ekkert er kommentað út

3.  Commenta út import Backgammon og kóða sem keyrir upp þjálfun í neural_network_agent.py

```py
 # import Backgammon.py

 ...

 Frá línu 179:
 """
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
training_steps = 50000
learnit(training_steps, epsilon, lam, alpha, V, alpha1, alpha2, w1, b2, w2, b2)
end = time.time()
print(end - start)

torch.save(w1, './w1_trained.pth')
torch.save(w2, './w2_trained.pth')
torch.save(b1, './b1_trained.pth')
torch.save(b2, './b2_trained.pth')
print('w1 from file',torch.load('./w1_trained.pth', map_location=lambda storage, loc: storage))
print('w2 from file',torch.load('./w2_trained.pth', map_location=lambda storage, loc: storage))
print('b1 from file',torch.load('./b1_trained.pth', map_location=lambda storage, loc: storage))
print('b2 from file',torch.load('./b2_trained.pth', map_location=lambda storage, loc: storage))
 """
```

4. Virkja aftur import agent.py í Backgammon.py

```py
import agent
```

5. Keyra Backgammon.py með

```
 if player == 1:
    move = agent.action(board_copy,dice,player,i)
 elif player == -1:
    move = random_agent(board_copy,dice,player,i)
```

í play_a_game.

## Nöfn og notendanöfn allra í hóp.

- Sigrún Dís Hauksdóttir - sdh20
- Ásta Lára Magnúsdóttir - alm20
- Hinrik Már Rögnvaldsson - hmr ..
- Marín Ingibjörg McGinley - mim ..

## The board interpretation

The game is set up in the **Backgammon.py** file. To play a game, simply run the program.
The two players are defined as player 1 and player -1.
The board has 29 positions:

- positions 1-24 are the labeled on-board positions
- positions 25 and 26 are the jails for when a piece is "killed" (25 is the jail for player 1 and 26 for player -1)
- positions 27 and 28 represent the position of the pieces who have been beard off (27 for player 1 and 28 for player -1)
- position 0 is pointless and is not being used...

The number of pieces in a certain possition of the board is represented by n where |n| is the number of pieces in the
position and sign(n) indicates which player owns the pieces in the position.

few examples:

- `board[23] = 3` means that player 1 has 3 player on the 23rd position of the board.
- `board[21] = -10` means that player -1 has 10 pieces on the 21st position of the board.
- `board[28] = -2` would mean that player -1 has beared off two pieces.

## Moving the pieces

The game is played between agents. Your main agent should be coded and trained in the **agent.py** file.
When the Backgammon.py program is executed, it imports your agent and uses his decisions to make moves.
The moves are simple. They are written as lists of two numbers where the

- first number represents the position from where you wish to move your piece from
- and the second number represents the position to where you wish to move your piece

When an agents is to move, it returns a couple of moves since he rolls two dices and therefore has to make two moves.
If there are less then 2 moves available (1 or 0) it can return fewer moves.

When a player rolls the same number on the dice in Backgammon, he is allowed to play 2 times. That is 2 moves 2 times.
When that happens, the agent should not return 4 moves in one. Instead, the **Backgammon.py** file asks the agent two times to make his move.

To decide on which move to make feel free to use the functions _legal_moves_, _legal_move_ and _update_board_ as you wish as well as making your own versions of them.

example:
to make the following move: http://www.bkgm.com/faq/gif/pickandpass.gif
the agent of player 1 has to return the list `[(10,5),(5,3)]`

## Thoughts and advices

### Running time

running time for one game is ~55ms per game or just under a minute per 1000 games when the players are only random agents that are not training. When training your agents, you might want to think cautiously about the time complexity of your code. Feel free to make your own faster code of Backgammon (and then share it!) but make sure your agents will be integrable for this version.

### Different perspectives

Your players have to be able to both play as player 1 and player -1. For this to be possible you can either

- flip the board and always make your player feel like player one. The code has already been made in the file **flipped_agent.py**. There you can find the functions _flip_board_ and _flip_move_ as well as an example of an agent that uses them correctly.
- account for both cases (as the moves will be different for the different players). Note that the training time will be twice as much as for the other option.

### Cheating

To save running time, the coded game doesn't check if your player is cheating. Because of that, you have to be careful that your player isn't doing so (e.g. accidentally making 7 moves instead of 2 or maybe moving his piece from the starting position to the end position and therefore always beating his opponents). The moves from the agents will be checked before they are submitted in the final tournament.
