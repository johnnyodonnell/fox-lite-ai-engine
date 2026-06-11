// Dump random full-match traces from the authoritative JS rules (src/engine/game.js)
// so the Rust port (foxlite_core) can replay the exact deals + moves and assert
// identical trick/score/winner outcomes.
//
//   node training/scripts/dump_rules_traces.mjs [numGames] > training/foxlite_core/tests/rules_traces.json
//
// Each round records its deal (hands/trump/leader/start-score), the ordered moves,
// the resulting tricksWon, and the cumulative score after end-of-round scoring.

import {
  createGame,
  legalMoves,
  playCard,
  advanceAfterTrick,
  endRound,
  HUMAN,
  BOT,
} from '../../src/engine/game.js'

function matchWinner(score) {
  // Tie-break: Human wins if human >= 21 and human >= bot, else Bot.
  if (score.human >= 21 && score.human >= score.bot) return HUMAN
  return BOT
}

function recordGame() {
  let state = createGame()
  const rounds = []
  while (state.phase !== 'match-over') {
    const round = {
      roundNum: state.roundNum,
      score: { ...state.score },
      humanHand: state.humanHand.map((c) => c.id),
      botHand: state.botHand.map((c) => c.id),
      trump: state.trump.id,
      leader: state.leader,
      moves: [],
    }
    while (state.phase !== 'round-over' && state.phase !== 'match-over') {
      if (state.phase === 'trick-complete') {
        state = advanceAfterTrick(state)
        continue
      }
      const hand = state.awaiting === HUMAN ? state.humanHand : state.botHand
      const legal = legalMoves(hand, state.ledCard)
      const card = legal[Math.floor(Math.random() * legal.length)]
      round.moves.push({ player: state.awaiting, card: card.id })
      state = playCard(state, card)
    }
    round.tricksWon = { ...state.tricksWon }
    const after = endRound(state)
    round.scoreAfter = { ...after.score }
    rounds.push(round)
    state = after
  }
  return { rounds, finalScore: { ...state.score }, winner: matchWinner(state.score) }
}

const numGames = parseInt(process.argv[2] || '300', 10)
const games = []
for (let i = 0; i < numGames; i++) games.push(recordGame())
process.stdout.write(JSON.stringify({ games }))
