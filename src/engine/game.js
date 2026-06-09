// Pure Fox in the Forest Lite rules — no React, no DOM.
// "Lite" = the standard rules with every odd-rank special ability removed.
// Cards are plain trick-takers; trump still applies.

export const SUITS = ['bells', 'keys', 'moons']
export const RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
export const HUMAN = 'human'
export const BOT = 'bot'
export const TARGET_SCORE = 21
export const TRICKS_PER_ROUND = 13

const SUIT_ORDER = Object.fromEntries(SUITS.map((s, i) => [s, i]))

export function cardId(suit, rank) {
  return `${suit}-${rank}`
}

function createDeck() {
  const deck = []
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      deck.push({ suit, rank, id: cardId(suit, rank) })
    }
  }
  return deck
}

function shuffle(deck) {
  const a = deck.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

export function sortHand(hand) {
  return hand.slice().sort((a, b) => {
    const s = SUIT_ORDER[a.suit] - SUIT_ORDER[b.suit]
    return s !== 0 ? s : a.rank - b.rank
  })
}

// Round 1 = human leads; rounds alternate thereafter.
function initialLeaderFor(roundNum) {
  return roundNum % 2 === 1 ? HUMAN : BOT
}

function dealRound(roundNum, score) {
  const shuffled = shuffle(createDeck())
  const humanHand = sortHand(shuffled.slice(0, 13))
  const botHand = sortHand(shuffled.slice(13, 26))
  const trump = shuffled[26]
  const leader = initialLeaderFor(roundNum)
  return {
    humanHand,
    botHand,
    trump,
    leader,
    ledCard: null,
    awaiting: leader,
    tricksWon: { human: 0, bot: 0 },
    score,
    roundNum,
    trickNum: 1,
    phase: 'playing',
    lastTrick: null,
    // Full per-round play history, in order. `lastTrick` is for the UI's
    // single-trick replay; `trickHistory` is what the AI engine consumes.
    trickHistory: [],
  }
}

export function createGame() {
  return dealRound(1, { human: 0, bot: 0 })
}

export function legalMoves(hand, ledCard) {
  if (!ledCard) return hand
  const sameSuit = hand.filter((c) => c.suit === ledCard.suit)
  return sameSuit.length > 0 ? sameSuit : hand
}

// Returns 'lead' or 'follow'.
export function trickWinner(ledCard, followCard, trumpSuit) {
  const leadIsTrump = ledCard.suit === trumpSuit
  const followIsTrump = followCard.suit === trumpSuit
  if (leadIsTrump && !followIsTrump) return 'lead'
  if (!leadIsTrump && followIsTrump) return 'follow'
  // Same suit (either both trump or both the led suit, or follow failed
  // to follow and didn't trump — in which case lead wins by default).
  if (followCard.suit !== ledCard.suit) return 'lead'
  return followCard.rank > ledCard.rank ? 'follow' : 'lead'
}

// Returns points awarded for that many tricks in a round (Lite scoring).
export function scoreForTricks(n) {
  if (n <= 3) return 6
  if (n === 4) return 1
  if (n === 5) return 2
  if (n === 6) return 3
  if (n <= 9) return 6
  return 0
}

function removeCard(hand, card) {
  return hand.filter((c) => c.id !== card.id)
}

function playerKey(player) {
  return player === HUMAN ? 'human' : 'bot'
}

function otherPlayer(player) {
  return player === HUMAN ? BOT : HUMAN
}

// Apply a single card play. Caller is responsible for only calling this
// when state.awaiting is the player who owns `card`.
export function playCard(state, card) {
  const player = state.awaiting
  const handKey = player === HUMAN ? 'humanHand' : 'botHand'
  const newHand = removeCard(state[handKey], card)
  const playEvent = { trick: state.trickNum, player, card }
  const trickHistory = [...state.trickHistory, playEvent]

  // Leading the trick.
  if (state.ledCard === null) {
    return {
      ...state,
      [handKey]: newHand,
      ledCard: card,
      awaiting: otherPlayer(player),
      trickHistory,
    }
  }

  // Following — resolve the trick.
  const winnerSide = trickWinner(state.ledCard, card, state.trump.suit)
  const winner = winnerSide === 'lead' ? state.leader : player
  const tricksWon = {
    ...state.tricksWon,
    [playerKey(winner)]: state.tricksWon[playerKey(winner)] + 1,
  }
  return {
    ...state,
    [handKey]: newHand,
    ledCard: null,
    awaiting: null,
    leader: winner,
    tricksWon,
    phase: 'trick-complete',
    lastTrick: {
      leadCard: state.ledCard,
      followCard: card,
      leader: state.leader,
      winner,
    },
    trickHistory,
  }
}

// Called by the UI after the brief pause showing the completed trick.
export function advanceAfterTrick(state) {
  const nextTrickNum = state.trickNum + 1
  if (nextTrickNum > TRICKS_PER_ROUND) {
    return {
      ...state,
      lastTrick: null,
      trickNum: nextTrickNum,
      awaiting: null,
      phase: 'round-over',
    }
  }
  return {
    ...state,
    lastTrick: null,
    trickNum: nextTrickNum,
    awaiting: state.leader,
    phase: 'playing',
  }
}

// Apply round-end scoring; either start the next round or end the match.
export function endRound(state) {
  const humanPts = scoreForTricks(state.tricksWon.human)
  const botPts = scoreForTricks(state.tricksWon.bot)
  const newScore = {
    human: state.score.human + humanPts,
    bot: state.score.bot + botPts,
  }
  if (newScore.human >= TARGET_SCORE || newScore.bot >= TARGET_SCORE) {
    return {
      ...state,
      score: newScore,
      awaiting: null,
      phase: 'match-over',
    }
  }
  return dealRound(state.roundNum + 1, newScore)
}

// The match winner, valid once phase is 'match-over'. Per the official rules, a
// tie on total points is broken in favor of whoever scored more in the final
// round (`state.tricksWon` still holds that round's result). Per-round point
// totals can never tie — the two sides split 13 tricks and scoreForTricks maps
// every such split to two different values — so this tie-break is always
// decisive.
export function matchWinner(state) {
  const { human, bot } = state.score
  if (human > bot) return HUMAN
  if (bot > human) return BOT
  const humanLast = scoreForTricks(state.tricksWon.human)
  const botLast = scoreForTricks(state.tricksWon.bot)
  return humanLast > botLast ? HUMAN : BOT
}

// Summary helper for the round-over banner.
export function roundSummary(state) {
  return {
    human: {
      tricks: state.tricksWon.human,
      points: scoreForTricks(state.tricksWon.human),
    },
    bot: {
      tricks: state.tricksWon.bot,
      points: scoreForTricks(state.tricksWon.bot),
    },
  }
}
