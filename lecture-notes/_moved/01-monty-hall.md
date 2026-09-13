Belongs in: Week 1 exercise session (exercise-sessions/01-introduction.qmd), as a warm-up
before the beta-binomial grid exercise.

Cut from lecture-notes/01-introduction.qmd (section "A puzzle that punishes intuition")
because chapter 1 now carries a single running example, the probability of a losing month.
Bayes' theorem for events stays in the chapter in general notation; only this second
example moves.

---

### A puzzle that punishes intuition

The classic test of whether one has understood the formula is the Monty Hall problem.
There are three doors, a prize behind one of them, and goats behind the other two. You
pick door 1. The host, who knows where the prize is, opens a different door that has a
goat behind it, say door 2, and offers you the chance to switch to door 3. Should you?

Most people say it cannot matter, because two doors remain. Bayes' theorem says
otherwise. Let $A$, $B$, $C$ be the events that the prize is behind door 1, 2, 3, each
with prior probability $1/3$. The evidence is that the host opened door 2. The
likelihoods of that evidence differ across the three explanations, and this is the entire
trick:

$$
P(\text{opens } 2 \mid A) = \tfrac{1}{2}, \qquad
P(\text{opens } 2 \mid B) = 0, \qquad
P(\text{opens } 2 \mid C) = 1 .
$$

If the prize is behind your door, the host may open either remaining door and picks door
2 half the time. If it is behind door 2 he cannot open it. If it is behind door 3 he has
no choice but to open door 2. Total probability gives
$P(\text{opens } 2) = \tfrac{1}{2}\cdot\tfrac{1}{3} + 0 \cdot \tfrac{1}{3} + 1 \cdot
\tfrac{1}{3} = \tfrac{1}{2}$, and therefore

$$
P(A \mid \text{opens } 2) = \frac{\tfrac{1}{2} \cdot \tfrac{1}{3}}{\tfrac{1}{2}}
= \frac{1}{3}, \qquad P(C \mid \text{opens } 2) = \frac{2}{3}.
$$

Switching doubles the chance of winning. The host's action was informative not because it
eliminated a door but because he was constrained by knowing the answer.

Since a derivation one does not believe is worth simulating, the next block plays the
game many times and counts. Each round draws the prize location and the player's choice,
lets the host open a goat door he is allowed to open, and records whether staying or
switching would have won.

**Reading the computation.** The simulated host obeys the same information rule used in the calculation. Count wins under staying and switching; this checks the model of the host, not a general claim that switching always helps.

```{python}
n_games = 200_000
prize = rng.integers(0, 3, n_games)          # door hiding the prize
choice = rng.integers(0, 3, n_games)         # player's first pick

# the host opens a door that is neither the player's pick nor the prize
doors = np.array([0, 1, 2])
allowed = (doors[None, :] != choice[:, None]) & (doors[None, :] != prize[:, None])
opened = (allowed * rng.random((n_games, 3))).argmax(axis=1)

switch_to = 3 - choice - opened              # the remaining door
print(f"win by staying:   {(choice == prize).mean():.4f}")
print(f"win by switching: {(switch_to == prize).mean():.4f}")
```

Staying wins about a third of the time and switching about two thirds, which is what the
posterior said. The point of the exercise is not the puzzle but the habit: we will write
down a posterior, then check it against a simulation that makes no use of the algebra.
