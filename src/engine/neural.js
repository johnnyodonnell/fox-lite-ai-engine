// The shipping AI engine — drop-in replacement for random.js.
//
//   bestMove(state) -> a card from state.botHand
//
// Currently a double-dummy engine: PIMC ensemble over an exact alpha-beta
// solver. See src/engine/dde.js and src/engine/doubleDummy.js.
//
// The previous neural-net engine (PIMC with a trained policy/value
// network) is preserved verbatim in src/engine/neuralNet.js so it's
// trivially revivable when we resume the AlphaZero path with a stronger
// teacher signal.

export { bestMove } from './dde.js'
