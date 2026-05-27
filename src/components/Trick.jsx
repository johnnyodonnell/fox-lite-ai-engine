import Card from './Card.jsx'

// Renders the two slots of the current trick. Empty slots are placeholders.
// When `winnerSide` is set ('lead' or 'follow'), that card is highlighted.
export default function Trick({ leadCard, followCard, leader, winnerSide }) {
  const leadLabel = leader === 'human' ? 'You' : 'Bot'
  const followLabel = leader === 'human' ? 'Bot' : 'You'

  return (
    <div className="trick">
      <TrickSlot
        label={leadLabel}
        roleLabel="led"
        card={leadCard}
        winner={winnerSide === 'lead'}
      />
      <TrickSlot
        label={followLabel}
        roleLabel="followed"
        card={followCard}
        winner={winnerSide === 'follow'}
      />
    </div>
  )
}

function TrickSlot({ label, roleLabel, card, winner }) {
  const classes = ['trick__slot']
  if (winner) classes.push('trick__slot--winner')
  return (
    <div className={classes.join(' ')}>
      <div className="trick__label">
        {label} {roleLabel}
      </div>
      {card ? <Card card={card} /> : <div className="card card--empty" />}
    </div>
  )
}
