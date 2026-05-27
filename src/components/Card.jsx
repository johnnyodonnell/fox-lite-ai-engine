const SUIT_ICONS = {
  bells: (
    <svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor" aria-hidden="true">
      <path d="M12 2a1.2 1.2 0 0 1 1.2 1.2v.9A7 7 0 0 1 19 11v3.2l1.6 1.9a1 1 0 0 1-.77 1.65H4.17a1 1 0 0 1-.77-1.65L5 14.2V11a7 7 0 0 1 5.8-6.9v-.9A1.2 1.2 0 0 1 12 2zm-2 17h4a2 2 0 0 1-4 0z" />
    </svg>
  ),
  keys: (
    <svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor" aria-hidden="true">
      <path d="M21 10h-8.35A5.99 5.99 0 0 0 7 6c-3.31 0-6 2.69-6 6s2.69 6 6 6a5.99 5.99 0 0 0 5.65-4H13l2 2 2-2 2 2 4-4.05L21 10zm-14 5a3 3 0 1 1 0-6 3 3 0 0 1 0 6z" />
    </svg>
  ),
  moons: (
    <svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor" aria-hidden="true">
      <path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z" />
    </svg>
  ),
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
        {SUIT_ICONS[card.suit]}
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
