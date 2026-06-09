import { HUMAN } from '../engine/game.js'

export default function RoundBanner({ type, summary, score, winner, onContinue }) {
  if (type === 'round') {
    return (
      <div className="banner">
        <h2 className="banner__title">Round complete</h2>
        <table className="banner__table">
          <thead>
            <tr>
              <th></th>
              <th>Tricks</th>
              <th>Points</th>
              <th>Match</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>You</th>
              <td>{summary.human.tricks}</td>
              <td>+{summary.human.points}</td>
              <td>{score.human + summary.human.points}</td>
            </tr>
            <tr>
              <th>Bot</th>
              <td>{summary.bot.tricks}</td>
              <td>+{summary.bot.points}</td>
              <td>{score.bot + summary.bot.points}</td>
            </tr>
          </tbody>
        </table>
        <button className="banner__btn" onClick={onContinue}>
          Next round
        </button>
      </div>
    )
  }

  // match-over. The winner is decided by the engine (matchWinner), which breaks
  // a tie on total points by the final round's points — so there is no draw.
  const title = winner === HUMAN ? 'You win the match!' : 'Bot wins the match'

  return (
    <div className="banner">
      <h2 className="banner__title">{title}</h2>
      <table className="banner__table">
        <tbody>
          <tr>
            <th>You</th>
            <td>{score.human}</td>
          </tr>
          <tr>
            <th>Bot</th>
            <td>{score.bot}</td>
          </tr>
        </tbody>
      </table>
      <button className="banner__btn" onClick={onContinue}>
        New match
      </button>
    </div>
  )
}
