export default function Status({ humanScore, botScore }) {
  return (
    <div className="status">
      <div className="status__row">
        <span className="status__chip status__chip--score">
          Match — You {humanScore} · Bot {botScore}
        </span>
      </div>
    </div>
  )
}
