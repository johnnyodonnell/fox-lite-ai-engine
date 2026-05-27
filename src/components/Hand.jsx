import Card from './Card.jsx'

export default function Hand({ cards, legalIds, onPlay, disabled }) {
  return (
    <div className="hand">
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
