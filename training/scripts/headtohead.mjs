// Head-to-head harness: play N matches between two engines and report win
// rates + per-round score margins. Used as the Phase 2 sanity gate
// (uniform-prior PIMC must crush random.js) and re-runnable as new engines
// land.
//
// Usage:
//   node training/scripts/headtohead.mjs <engineA.js> <engineB.js> [N]
//   node training/scripts/headtohead.mjs ../../src/engine/neural.js ../../src/engine/random.js 40
//
// Each engine file is expected to default-export nothing and named-export
// `bestMove(state) -> card from state.botHand`.

import path from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'

import {
  HUMAN,
  BOT,
  advanceAfterTrick,
  createGame,
  endRound,
  playCard,
} from '../../src/engine/game.js'

const here = path.dirname(fileURLToPath(import.meta.url))

function swapPlayer(p) {
  if (p === HUMAN) return BOT
  if (p === BOT) return HUMAN
  return p
}

// Flip the state's two seats so an engine written for "bot" can act as the
// human player. Card objects are identity-free, so swapping hands is a literal
// reassignment.
function swapSeats(state) {
  return {
    ...state,
    humanHand: state.botHand,
    botHand: state.humanHand,
    leader: swapPlayer(state.leader),
    awaiting: swapPlayer(state.awaiting),
    tricksWon: { human: state.tricksWon.bot, bot: state.tricksWon.human },
    score: { human: state.score.bot, bot: state.score.human },
    lastTrick: state.lastTrick && {
      ...state.lastTrick,
      leader: swapPlayer(state.lastTrick.leader),
      winner: swapPlayer(state.lastTrick.winner),
    },
    trickHistory: state.trickHistory.map((ev) => ({ ...ev, player: swapPlayer(ev.player) })),
  }
}

function callEngine(engine, state, seat) {
  if (seat === BOT) return engine(state)
  return engine(swapSeats(state))
}

function playOneMatch(engineBot, engineHuman) {
  let state = createGame()
  const roundMargins = [] // (bot points - human points) per round
  let priorScore = { human: 0, bot: 0 }
  while (state.phase !== 'match-over') {
    if (state.phase === 'round-over') {
      state = endRound(state)
      const dh = state.score.human - priorScore.human
      const db = state.score.bot - priorScore.bot
      roundMargins.push(db - dh)
      priorScore = { human: state.score.human, bot: state.score.bot }
      continue
    }
    if (state.phase === 'trick-complete') {
      state = advanceAfterTrick(state)
      continue
    }
    const seat = state.awaiting
    const engine = seat === BOT ? engineBot : engineHuman
    const card = callEngine(engine, state, seat)
    state = playCard(state, card)
  }
  return { score: state.score, roundMargins }
}

async function loadEngine(modulePath) {
  const url = pathToFileURL(path.resolve(here, modulePath)).href
  const mod = await import(url)
  if (typeof mod.bestMove !== 'function') {
    throw new Error(`module ${modulePath} has no named export 'bestMove'`)
  }
  return mod.bestMove
}

function summarize(label, wins, draws, total, margins) {
  const meanMargin = margins.reduce((a, b) => a + b, 0) / Math.max(margins.length, 1)
  const winRate = wins / total
  console.log(
    `  ${label.padEnd(8)} wins=${wins}  draws=${draws}  ` +
    `wr=${(winRate * 100).toFixed(1)}%  ` +
    `mean per-round margin (A−B)=${meanMargin.toFixed(2)}`
  )
}

async function main() {
  const [engineAPath, engineBPath, nArg] = process.argv.slice(2)
  if (!engineAPath || !engineBPath) {
    console.error('usage: node training/scripts/headtohead.mjs <engineA.js> <engineB.js> [N]')
    process.exit(2)
  }
  const N = parseInt(nArg || '40', 10)
  const engineA = await loadEngine(engineAPath)
  const engineB = await loadEngine(engineBPath)

  console.log(`Head-to-head: ${engineAPath} (A) vs ${engineBPath} (B) — ${N} matches`)
  console.log('  Seats swap halfway so each engine gets equal first-leader rounds.\n')

  let aWins = 0, bWins = 0, draws = 0
  const allAmarginsPerRound = []

  for (let i = 0; i < N; i++) {
    // First half: A is bot, B is human. Second half: roles swap.
    const aIsBot = i < N / 2
    const { score, roundMargins } = aIsBot
      ? playOneMatch(engineA, engineB)
      : playOneMatch(engineB, engineA)
    const aScore = aIsBot ? score.bot : score.human
    const bScore = aIsBot ? score.human : score.bot
    if (aScore > bScore) aWins++
    else if (bScore > aScore) bWins++
    else draws++
    // Re-orient round margins to be (A - B) regardless of seat.
    for (const m of roundMargins) allAmarginsPerRound.push(aIsBot ? m : -m)
    process.stdout.write(`\r  played ${i + 1}/${N} matches`)
  }
  console.log()
  console.log()
  summarize('A', aWins, draws, N, allAmarginsPerRound)
  summarize('B', bWins, draws, N, allAmarginsPerRound.map((m) => -m))
}

main().catch((e) => { console.error(e); process.exit(1) })
