import Card from './Card.jsx'

export default function Hand({ cards, legalIds, onPlay, disabled }) {
  // While it isn't the human's turn (e.g. waiting on the bot to submit its
  // move) the whole hand is locked: dimmed for feedback and pointer-events
  // disabled so a card can't be selected ahead of the opponent.
  return (
    <div className={disabled ? 'hand hand--locked' : 'hand'}>
      {cards.map((card) => {
        const isLegal = legalIds.has(card.id)
        const dimmed = !disabled && !isLegal
        return (
          <Card
            key={card.id}
            card={card}
            onClick={isLegal && !disabled ? onPlay : undefined}
            disabled={disabled || !isLegal}
            dimmed={dimmed}
          />
        )
      })}
    </div>
  )
}
