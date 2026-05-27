export default function Status({
  message,
  roundNum,
  trickNum,
  humanTricks,
  botTricks,
  humanScore,
  botScore,
}) {
  return (
    <div className="status">
      <div className="status__message">{message}</div>
      <div className="status__row">
        <span className="status__chip">Round {roundNum}</span>
        <span className="status__chip">Trick {trickNum}/13</span>
      </div>
      <div className="status__row">
        <span className="status__chip">
          Tricks — You {humanTricks} · Bot {botTricks}
        </span>
        <span className="status__chip status__chip--score">
          Match — You {humanScore} · Bot {botScore}
        </span>
      </div>
    </div>
  )
}
