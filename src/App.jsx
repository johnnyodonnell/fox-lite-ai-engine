import { useEffect, useState } from 'react'
import BotHand from './components/BotHand.jsx'
import Hand from './components/Hand.jsx'
import RoundBanner from './components/RoundBanner.jsx'
import Status from './components/Status.jsx'
import Trick from './components/Trick.jsx'
import Trump from './components/Trump.jsx'
import { bestMove } from './engine/dde.js'
import {
  BOT,
  HUMAN,
  advanceAfterTrick,
  createGame,
  endRound,
  legalMoves,
  playCard,
  roundSummary,
} from './engine/game.js'

const TRICK_PAUSE_MS = 1000

function statusMessage(state) {
  if (state.phase === 'trick-complete') {
    const winner = state.lastTrick.winner === HUMAN ? 'You take' : 'Bot takes'
    return `${winner} the trick`
  }
  if (state.phase === 'round-over') return 'Round complete'
  if (state.phase === 'match-over') return 'Match complete'
  if (state.awaiting === HUMAN) {
    return state.ledCard ? 'Your turn — follow' : 'Your turn — lead'
  }
  return 'Bot is playing…'
}

export default function App() {
  const [state, setState] = useState(createGame)

  // After a completed trick, pause ~1s then advance.
  useEffect(() => {
    if (state.phase !== 'trick-complete') return
    const id = setTimeout(() => {
      setState((s) => (s.phase === 'trick-complete' ? advanceAfterTrick(s) : s))
    }, TRICK_PAUSE_MS)
    return () => clearTimeout(id)
  }, [state.phase])

  // Bot plays instantly whenever it's the bot's turn.
  useEffect(() => {
    if (state.phase !== 'playing' || state.awaiting !== BOT) return
    setState((s) => {
      if (s.phase !== 'playing' || s.awaiting !== BOT) return s
      return playCard(s, bestMove(s))
    })
  }, [state.phase, state.awaiting])

  function handleHumanPlay(card) {
    setState((s) => {
      if (s.phase !== 'playing' || s.awaiting !== HUMAN) return s
      const legal = legalMoves(s.humanHand, s.ledCard)
      if (!legal.some((c) => c.id === card.id)) return s
      return playCard(s, card)
    })
  }

  function handleNextRound() {
    setState((s) => endRound(s))
  }

  function handleNewMatch() {
    setState(createGame())
  }

  const legal =
    state.phase === 'playing' && state.awaiting === HUMAN
      ? legalMoves(state.humanHand, state.ledCard)
      : []
  const legalIds = new Set(legal.map((c) => c.id))
  const handDisabled =
    state.phase !== 'playing' || state.awaiting !== HUMAN

  const trickDisplay =
    state.phase === 'trick-complete'
      ? {
          leadCard: state.lastTrick.leadCard,
          followCard: state.lastTrick.followCard,
          leader: state.lastTrick.leader,
          winnerSide:
            state.lastTrick.winner === state.lastTrick.leader
              ? 'lead'
              : 'follow',
        }
      : {
          leadCard: state.ledCard,
          followCard: null,
          leader: state.leader,
          winnerSide: null,
        }

  return (
    <main className="app">
      <h1>Fox Lite</h1>

      <Status
        message={statusMessage(state)}
        roundNum={state.roundNum}
        trickNum={Math.min(state.trickNum, 13)}
        humanTricks={state.tricksWon.human}
        botTricks={state.tricksWon.bot}
        humanScore={state.score.human}
        botScore={state.score.bot}
      />

      <BotHand count={state.botHand.length} />

      <div className="table">
        <Trump card={state.trump} />
        <Trick {...trickDisplay} />
      </div>

      <Hand
        cards={state.humanHand}
        legalIds={legalIds}
        onPlay={handleHumanPlay}
        disabled={handDisabled}
      />

      {state.phase === 'round-over' && (
        <RoundBanner
          type="round"
          summary={roundSummary(state)}
          score={state.score}
          onContinue={handleNextRound}
        />
      )}
      {state.phase === 'match-over' && (
        <RoundBanner
          type="match"
          score={state.score}
          onContinue={handleNewMatch}
        />
      )}
    </main>
  )
}
