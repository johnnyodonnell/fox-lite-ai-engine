"""Central configuration for the Fox-Lite AlphaZero pipeline.

Every tunable lives here. Values the browser engine also needs (cPuct,
sim counts, number of determinizations) are re-emitted into weights.json by
scripts/export_weights.py so the JS side stays in sync.
"""

# -- Reproducibility -------------------------------------------------------
SEED = 1234

# -- Network ---------------------------------------------------------------
HIDDEN_SIZE = 256
TRUNK_LAYERS = 3          # number of Linear+ReLU layers before the heads

# -- PIMC / MCTS -----------------------------------------------------------
C_PUCT = 1.5
# Training-time MCTS budget (per move, per determinization).
NUM_SIMULATIONS = 64
# How many determinizations to ensemble per move at training time.
NUM_DETERMINIZATIONS = 4

# Higher search budgets used at evaluation / browser-deploy time. Browser
# latency is unbounded per the project plan, so the eval bot searches deeper
# than self-play does.
PLAY_SIMULATIONS = 200
PLAY_DETERMINIZATIONS = 16

DIRICHLET_ALPHA = 0.5     # 33 cards but typically <= ~10 legal at a time
DIRICHLET_EPSILON = 0.25  # mixing weight (self-play only)

# -- Self-play -------------------------------------------------------------
TEMPERATURE_MOVES = 8     # first N plies of a round sampled with T=1, then argmax

# -- Training --------------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
TRAIN_STEPS_PER_ITER = 200

# -- Replay buffer ---------------------------------------------------------
BUFFER_SIZE = 200_000

# -- Training loop ---------------------------------------------------------
NUM_ITERATIONS = 40
GAMES_PER_ITER = 100      # matches per iter (each plays out >= 1 round)

# -- Arena (promotion gate) ------------------------------------------------
ARENA_GAMES = 30
ARENA_WIN_RATE = 0.55     # challenger needs >= this to replace best.pt

# -- Safety tripwires ------------------------------------------------------
MAX_WEIGHT_NORM = 1e4     # halt training if any param's L2 norm blows past this
