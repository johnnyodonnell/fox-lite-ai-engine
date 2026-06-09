export default function Status({
  message,
  roundNum,
  humanScore,
  botScore,
}) {
  return (
    <div className="status">
      <div className="status__message">{message}</div>
      <div className="status__row">
        <span className="status__chip">Round {roundNum}</span>
        <span className="status__chip status__chip--score">
          Match — You {humanScore} · Bot {botScore}
        </span>
      </div>
    </div>
  )
}
