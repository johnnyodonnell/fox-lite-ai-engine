export default function RoundBanner({ type, summary, score, onContinue }) {
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

  // match-over
  const winner =
    score.human > score.bot
      ? 'You win the match!'
      : score.bot > score.human
        ? 'Bot wins the match'
        : "It's a tie"

  return (
    <div className="banner">
      <h2 className="banner__title">{winner}</h2>
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
