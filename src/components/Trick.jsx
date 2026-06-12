import Card from './Card.jsx'

// Renders the two slots of the current trick, staggered: the bot's card
// always on the left and raised, the player's always on the right and
// lowered. Empty slots are placeholders. When `winnerSide` is set ('lead'
// or 'follow'), that card is highlighted.
export default function Trick({ leadCard, followCard, leader, winnerSide }) {
  const botSide = leader === 'human' ? 'follow' : 'lead'
  const humanSide = leader === 'human' ? 'lead' : 'follow'

  return (
    <div className="trick">
      <TrickSlot
        card={botSide === 'lead' ? leadCard : followCard}
        winner={winnerSide === botSide}
      />
      <TrickSlot
        card={humanSide === 'lead' ? leadCard : followCard}
        winner={winnerSide === humanSide}
      />
    </div>
  )
}

function TrickSlot({ card, winner }) {
  const classes = ['trick__slot']
  if (winner) classes.push('trick__slot--winner')
  return (
    <div className={classes.join(' ')}>
      {card ? <Card card={card} /> : <div className="card card--empty" />}
    </div>
  )
}
