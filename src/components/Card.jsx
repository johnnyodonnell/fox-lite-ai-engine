const SUIT_SYMBOLS = {
  bells: '🔔',
  keys: '🗝️',
  moons: '🌙',
}

export default function Card({ card, faceDown, onClick, disabled, dimmed }) {
  if (faceDown || !card) {
    return <div className="card card--face-down" />
  }

  const classes = ['card', `card--suit-${card.suit}`]
  if (onClick) classes.push('card--clickable')
  if (dimmed) classes.push('card--dimmed')

  const inner = (
    <>
      <span className="card__rank">{card.rank}</span>
      <span className={`card__suit suit-${card.suit}`}>
        {SUIT_SYMBOLS[card.suit]}
      </span>
    </>
  )

  if (onClick) {
    return (
      <button
        type="button"
        className={classes.join(' ')}
        onClick={() => onClick(card)}
        disabled={disabled}
        aria-label={`${card.rank} of ${card.suit}`}
      >
        {inner}
      </button>
    )
  }

  return (
    <div className={classes.join(' ')} aria-label={`${card.rank} of ${card.suit}`}>
      {inner}
    </div>
  )
}
