import Card from './Card.jsx'

export default function Trump({ card }) {
  return (
    <div className="trump">
      <div className="trump__label">Decree (trump)</div>
      <Card card={card} />
    </div>
  )
}
