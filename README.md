## Connect6 AI based on reinforcement learning

Connect6 AI program using VCDT search and REINFORCE algorithm
The uploaded model wins the NCTU6 level 2 ([NCTU6 Android App](https://play.google.com/store/apps/details?id=tw.edu.nctu.csie.cyclab.connect6))



## Installation

python 3.x, tensorflow, keras, numpy, h5py

```
pip install -r requirements.txt
```



## Usage

#### Train

A system to train connect6 AI.

```
python train.py
```

##### Configuration (train.json)

**checkpoint_period** (int): This value represents agent save interval. Evaluation is conducted before the agent is saved, and the agent may be sampled later (See Training Flow).
**sampling_range** (int): This value is used to specify how many top agents should be considered when sampling agent for each iteration (See Training Flow).



#### Test

A system in which a lot of agents can play against each other and obtain results.

```
python test.py
```

##### Configuration (test.json)

**black_checkpoint** (list): This list contains agent versions that want to play with black. If this value is ```null```, all versions of agent participate in the game.
**white_checkpoint** (list): This list contains agent versions that want to play with white. If this value is ```null```, all versions of agent participate in the game.



#### Play

A system that can play connect6 game visually. Not only can both agents play against each other, but it is also possible to play with human. When playing a game with human, you can place a stone with a mouse click.

```
python play.py
```

##### Configuration (play.json)

**black_checkpoint** (int): This value is agent version that want to play with black. If this value is ```null```, it is possible to play with human.
**white_checkpoint** (int): This value is agent version that want to play with white. If this value is ```null```, it is possible to play with human.
**board** (list): You can specify the state of game board (See example json files).



## Methods

#### Algorithm Flow

 <img src="asset/algorithm flow.png" height="350px"></img>



##### VCDT

> A type of winning strategy, called Victory by Continuous Double-Threat-or-more moves.

In this algorithm, only the top N actions from neural network are searched using level-synchronized parallel BFS.



#### Training Flow

 <img src="asset/training flow.png" height="250px"></img>



**Sampling method**: Extract randomly black and white agents from top N agents.
**Evaluation method**: Measure the winning rate by playing against all saved agents.
**Training method**: Use REINFORCE algorithm.



#### Other details

**Restrict moves**

- Place stones within three spaces of the diagonal, up and down, left and right, based on the stone placed on the board.
- If player can now connect six stones in a line, connect them.
- If opponent can connect six stones in a line on the next turn, defend it.



## Results

 <img src="asset/agent league results (plot).png" height="400px"></img>

 <img src="asset/agent league results (heatmap).png" height="450px"></img>



## Todo

- Fixing a potential bug that doing different action depending on completion order of threads in VCDT search.
- Solving the issue that requires more time for evaluation as agent version increase.
- Combining AlphaZero method and domain knowledge (e.g. VCDT).