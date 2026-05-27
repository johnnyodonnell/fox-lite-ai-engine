import Card from './Card.jsx'

export default function BotHand({ count }) {
  return (
    <div className="hand hand--bot">
      {Array.from({ length: count }, (_, i) => (
        <Card key={i} faceDown />
      ))}
    </div>
  )
}
