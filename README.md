# Fox in the Forest Lite — AI Engine

A 2-player browser version of *Fox in the Forest* (React + Vite) with a
swappable AI engine.

The "Lite" variant strips every special ability from the odd-rank cards
(1/3/5/7/9/11) — they are plain cards. Trump and trick-taking work exactly
like the original game.

The v1 AI engine is **random**: it picks a uniform-random legal card every
turn. The engine module exposes a single `bestMove(state)` function, so a
smarter engine (search, neural net, remote service) can replace it later
without touching the UI.

## Running locally

```sh
npm install
npm run dev        # or: ./run-local.sh
```

Then open the printed URL. `npm run build` produces a production build in `dist/`.

## Rules (Lite)

- **Deck**: 33 cards — 3 suits (🔔 Bells, 🗝️ Keys, 🌙 Moons) × ranks 1–11.
- **Deal**: 13 cards each; one card is revealed as the trump (decree). The
  remaining 6 are unused.
- **Play**: leader plays any card; follower must follow suit if able,
  otherwise any card (including trump).
- **Trick winner**: highest card of the led suit, unless trumped (then
  highest trump). Winner leads the next trick.
- **Round scoring** (after 13 tricks):

  | Tricks taken | Points |
  | --- | --- |
  | 0–3 | 6 |
  | 4 | 1 |
  | 5 | 2 |
  | 6 | 3 |
  | 7–9 | 6 |
  | 10–13 | 0 |

- **Match**: first to **21** points wins. The leader of trick 1 alternates
  between rounds (human leads round 1, bot leads round 2, etc.).

## Project layout

```
src/
  main.jsx              entry — mounts <App>
  App.jsx               game state + turn flow
  styles/app.css        single global stylesheet
  engine/
    game.js             pure Lite rules — no React, no side effects
    random.js           v1 AI — bestMove(state) returns a random legal card
  components/
    Card.jsx            a single card (rank + suit emoji, colored)
    Hand.jsx            the human's hand (clickable, sorted)
    BotHand.jsx         the opponent's hand (face-down count only)
    Trick.jsx           the current trick (lead card + follow card)
    Trump.jsx           the decree card display
    Status.jsx          turn message + tricks-won + match score
    RoundBanner.jsx     end-of-round / end-of-match overlay
```

## Swapping the engine

`src/App.jsx` imports `bestMove` from exactly one engine file. To plug in
a different engine, change that one line:

```js
import { bestMove } from './engine/random.js'    // current
// import { bestMove } from './engine/<other>.js'
```

The engine contract:

```js
bestMove(state)  // state: game state (see engine/game.js)  ->  Card from state.botHand
```
