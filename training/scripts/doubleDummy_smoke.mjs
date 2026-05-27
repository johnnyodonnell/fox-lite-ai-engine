// Smoke tests for src/engine/doubleDummy.js.
//
// 1. Last-card endgame: single forced play; assert the exact margin.
// 2. Last-two-trick toy: solver value must match a brute-force minimax
//    reference that has no TT and no collapsing — exercises both
//    optimizations.
// 3. A batch of random small positions (≤ 5 cards each side) all checked
//    against the brute-force reference. Catches any divergence between the
//    optimized and naive paths, including TT-bound-handling bugs.
//
// Run:  node training/scripts/doubleDummy_smoke.mjs

import {
  BOT,
  HUMAN,
  SUITS,
  TRICKS_PER_ROUND,
  advanceAfterTrick,
  cardId,
  legalMoves,
  playCard,
  scoreForTricks,
} from '../../src/engine/game.js'
import { solve } from '../../src/engine/doubleDummy.js'

// -- World construction helpers -------------------------------------------

function card(suit, rank) {
  return { suit, rank, id: cardId(suit, rank) }
}

// Build a `playing`-phase world from compact specs. Caller specifies
// hands and the bookkeeping; everything not relevant to the solver gets
// neutral defaults.
function world({
  humanHand, botHand, trump, leader = BOT, ledCard = null,
  tricksWonBot = 0, tricksWonHuman = 0,
}) {
  const trickNum = tricksWonBot + tricksWonHuman + 1
  return {
    humanHand,
    botHand,
    trump,
    leader,
    ledCard,
    awaiting: ledCard === null ? leader : otherOf(leader),
    tricksWon: { human: tricksWonHuman, bot: tricksWonBot },
    score: { human: 0, bot: 0 },
    roundNum: 1,
    trickNum,
    phase: 'playing',
    lastTrick: null,
    trickHistory: [],
  }
}
function otherOf(p) { return p === BOT ? HUMAN : BOT }

// -- Brute-force reference: naive minimax, no TT, no collapsing -----------

function bruteSolve(w, rootMover) {
  if (w.phase === 'round-over') {
    const b = scoreForTricks(w.tricksWon.bot)
    const h = scoreForTricks(w.tricksWon.human)
    const bf = b - h
    return rootMover === BOT ? bf : -bf
  }
  const mover = w.awaiting
  const handKey = mover === HUMAN ? 'humanHand' : 'botHand'
  const moves = legalMoves(w[handKey], w.ledCard)
  let best = mover === rootMover ? -Infinity : Infinity
  for (const m of moves) {
    let next = playCard(w, m)
    while (next.phase === 'trick-complete') next = advanceAfterTrick(next)
    const v = bruteSolve(next, rootMover)
    if (mover === rootMover) {
      if (v > best) best = v
    } else {
      if (v < best) best = v
    }
  }
  return best
}

// -- Test 1: last-card endgame, known margin -----------------------------

function test1_lastCardKnown() {
  // BOT leads trick 13. BOT has bells-5; opp has bells-3. BOT wins trick.
  // Final tricks: bot=7, human=6. Scores: 6 vs 3. Margin (bot frame) = +3.
  const w = world({
    humanHand: [card('bells', 3)],
    botHand: [card('bells', 5)],
    trump: card('moons', 11),
    leader: BOT,
    tricksWonBot: 6, tricksWonHuman: 6,
  })
  const { value, bestMove } = solve(w, -Infinity, +Infinity, new Map(), BOT)
  const ok = value === 3 && bestMove === 'bells-5'
  console.log(`  ${ok ? 'OK  ' : 'FAIL'}  last-card forced  value=${value} (expect +3)  bestMove=${bestMove}`)
  return ok ? 0 : 1
}

// -- Test 2: scoring-curve trap ------------------------------------------

function test2_scoringTrap() {
  // 2 tricks left. BOT at 3 tricks, opp at 9. The "winner-take-all" line
  // sends BOT to 5 (= 2 pts) and opp to 9 (= 6 pts) for margin -4. But if
  // BOT can ARRANGE to lose both and stay at 3, that's 6 pts for BOT and
  // opp goes to 11 (= 0 pts) for margin +6.
  //
  // Whether the trap is reachable depends on the hands; this test only
  // verifies that the optimized solver and the brute-force minimax agree
  // on the value. Acts as a meaningful end-to-end correctness check
  // because the position has real strategic content from the non-monotonic
  // scoring.
  const w = world({
    humanHand: [card('bells', 3), card('bells', 9)],
    botHand: [card('bells', 5), card('moons', 1)],
    trump: card('moons', 11),
    leader: BOT,
    tricksWonBot: 3, tricksWonHuman: 9,
  })
  const { value: vOpt } = solve(w, -Infinity, +Infinity, new Map(), BOT)
  const vRef = bruteSolve(w, BOT)
  const ok = vOpt === vRef
  console.log(`  ${ok ? 'OK  ' : 'FAIL'}  scoring-trap  optimized=${vOpt}  brute=${vRef}`)
  return ok ? 0 : 1
}

// -- Test 3: random small positions vs brute force -----------------------

function shuffleInPlace(a, rng) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
}

function allCards() {
  const out = []
  for (const s of SUITS) for (let r = 1; r <= 11; r++) out.push(card(s, r))
  return out
}

function test3_randomVsBrute() {
  let seed = 1
  const rng = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed / 0x100000000
  }
  let failures = 0
  let runs = 0
  for (let trial = 0; trial < 30; trial++) {
    // small hands: 3 cards each
    const HAND_SIZE = 3
    const deck = allCards()
    shuffleInPlace(deck, rng)
    const trump = deck[0]
    const botHand = deck.slice(1, 1 + HAND_SIZE)
    const humanHand = deck.slice(1 + HAND_SIZE, 1 + 2 * HAND_SIZE)
    const leader = rng() < 0.5 ? BOT : HUMAN
    // Total tricks completed must equal 13 − HAND_SIZE for the state to be
    // consistent (each player played one card per completed trick).
    const completed = TRICKS_PER_ROUND - HAND_SIZE
    const tricksWonBot = Math.floor(rng() * (completed + 1))
    const tricksWonHuman = completed - tricksWonBot
    const w = world({
      humanHand, botHand, trump, leader,
      tricksWonBot, tricksWonHuman,
    })
    const vOpt = solve(w, -Infinity, +Infinity, new Map(), BOT).value
    const vRef = bruteSolve(w, BOT)
    if (vOpt !== vRef) {
      failures++
      if (failures <= 3) {
        console.log(`  FAIL  random[${trial}] optimized=${vOpt} brute=${vRef}`)
        console.log(`         botHand=${JSON.stringify(botHand.map(c=>c.id))}`)
        console.log(`         humanHand=${JSON.stringify(humanHand.map(c=>c.id))}`)
        console.log(`         trump=${trump.id} leader=${leader} won=(${tricksWonBot},${tricksWonHuman})`)
      }
    }
    runs++
  }
  console.log(`  ${failures === 0 ? 'OK  ' : 'FAIL'}  random vs brute  ${runs - failures}/${runs} cases match`)
  return failures
}

// -- Test 4: TT-bound exercise (specific alpha/beta windows) -------------

function test4_ttBounds() {
  // Same position solved with various narrow windows. The cached values
  // from a tight window must not corrupt subsequent searches at wider
  // windows (or vice versa). We solve with full window, then again from
  // scratch through a narrow window, asserting the full-window result
  // still matches brute force.
  // 3 cards each + 10 completed tricks (5 each) → consistent.
  const w = world({
    humanHand: [card('bells', 3), card('bells', 9), card('keys', 7)],
    botHand: [card('bells', 5), card('moons', 1), card('keys', 10)],
    trump: card('moons', 11),
    leader: BOT,
    tricksWonBot: 5, tricksWonHuman: 5,
  })
  const truth = bruteSolve(w, BOT)
  // Run with several narrow windows; cache results in the same TT.
  const tt = new Map()
  for (const [a, b] of [[-1, 1], [0, 1], [-2, 2], [-Infinity, +Infinity]]) {
    solve(w, a, b, tt, BOT)
  }
  // Then ask the full-window question again.
  const { value } = solve(w, -Infinity, +Infinity, tt, BOT)
  const ok = value === truth
  console.log(`  ${ok ? 'OK  ' : 'FAIL'}  TT-bound shared cache  value=${value} truth=${truth}`)
  return ok ? 0 : 1
}

// ------------------------------------------------------------------------

let total = 0
total += test1_lastCardKnown()
total += test2_scoringTrap()
total += test3_randomVsBrute()
total += test4_ttBounds()

console.log()
if (total === 0) {
  console.log('SMOKE OK — doubleDummy.js passes all sanity checks')
  process.exit(0)
} else {
  console.log(`SMOKE FAILED — ${total} case(s) wrong`)
  process.exit(1)
}
