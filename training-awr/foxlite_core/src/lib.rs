//! Fox in the Forest **Lite** rules — a faithful port of `src/engine/game.js`.
//!
//! "Lite" = the standard rules with every odd-rank special ability removed.
//! Cards are plain trick-takers; trump still applies. Two players, a 33-card
//! deck (3 suits x ranks 1..=11), 13 cards per hand, 1 trump revealed, 6 unused.
//! Match is first to 21 points; round-1 leader is `Human`, alternating after.
//!
//! This crate is pure rules + (in `encode.rs`) the canonical NN input encoding.
//! No torch, no I/O. RNG is always passed in so games are reproducible.

use rand::Rng;

pub mod encode;

pub const NUM_SUITS: usize = 3;
pub const NUM_RANKS: usize = 11;
pub const NUM_CARDS: usize = NUM_SUITS * NUM_RANKS; // 33
pub const TARGET_SCORE: u32 = 21;
pub const TRICKS_PER_ROUND: u32 = 13;

/// Suit ordering matches `game.js`: bells=0, keys=1, moons=2.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    Human = 0,
    Bot = 1,
}

impl Player {
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    #[inline]
    pub fn other(self) -> Player {
        match self {
            Player::Human => Player::Bot,
            Player::Bot => Player::Human,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Playing,
    TrickComplete,
    RoundOver,
    MatchOver,
}

/// A card. `suit` in 0..3, `rank` in 1..=11.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Card {
    pub suit: u8,
    pub rank: u8,
}

impl Card {
    #[inline]
    pub fn new(suit: u8, rank: u8) -> Card {
        debug_assert!((suit as usize) < NUM_SUITS);
        debug_assert!(rank >= 1 && (rank as usize) <= NUM_RANKS);
        Card { suit, rank }
    }
    /// Canonical card index, matching `game.js` deck order (suit-outer,
    /// rank-inner): bells-1=0 .. moons-11=32.
    #[inline]
    pub fn index(self) -> usize {
        self.suit as usize * NUM_RANKS + (self.rank as usize - 1)
    }
    #[inline]
    pub fn from_index(i: usize) -> Card {
        debug_assert!(i < NUM_CARDS);
        Card {
            suit: (i / NUM_RANKS) as u8,
            rank: (i % NUM_RANKS) as u8 + 1,
        }
    }
}

/// One half-move in the current round's history (resets each round).
#[derive(Clone, Copy, Debug)]
pub struct PlayEvent {
    pub trick: u32,
    pub player: Player,
    pub card: Card,
}

/// Which side of the current trick won.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Lead,
    Follow,
}

#[derive(Clone, Debug)]
pub struct State {
    pub human_hand: Vec<Card>,
    pub bot_hand: Vec<Card>,
    pub trump: Card,
    pub leader: Player,
    pub led_card: Option<Card>,
    pub awaiting: Option<Player>,
    pub tricks_won: [u32; 2],
    pub score: [u32; 2],
    pub round_num: u32,
    pub trick_num: u32,
    pub phase: Phase,
    pub trick_history: Vec<PlayEvent>,
}

/// All 33 cards in canonical order.
pub fn create_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(NUM_CARDS);
    for suit in 0..NUM_SUITS as u8 {
        for rank in 1..=NUM_RANKS as u8 {
            deck.push(Card::new(suit, rank));
        }
    }
    deck
}

/// Fisher-Yates shuffle (mirrors the JS shuffle direction; the RNG sequence is
/// not expected to match Math.random — parity tests replay recorded deals).
fn shuffle<R: Rng + ?Sized>(deck: &mut [Card], rng: &mut R) {
    for i in (1..deck.len()).rev() {
        let j = rng.gen_range(0..=i);
        deck.swap(i, j);
    }
}

/// Sort by (suit, rank), matching `sortHand` in game.js.
pub fn sort_hand(hand: &mut [Card]) {
    hand.sort_by(|a, b| (a.suit, a.rank).cmp(&(b.suit, b.rank)));
}

/// Round 1 = Human leads; rounds alternate thereafter.
pub fn initial_leader_for(round_num: u32) -> Player {
    if round_num % 2 == 1 {
        Player::Human
    } else {
        Player::Bot
    }
}

/// Legal moves for a hand given the led card (must follow suit if able).
pub fn legal_moves(hand: &[Card], led_card: Option<Card>) -> Vec<Card> {
    match led_card {
        None => hand.to_vec(),
        Some(led) => {
            let same: Vec<Card> = hand.iter().copied().filter(|c| c.suit == led.suit).collect();
            if same.is_empty() {
                hand.to_vec()
            } else {
                same
            }
        }
    }
}

/// Trick resolution — returns which side won.
pub fn trick_winner(led: Card, follow: Card, trump_suit: u8) -> Side {
    let lead_is_trump = led.suit == trump_suit;
    let follow_is_trump = follow.suit == trump_suit;
    if lead_is_trump && !follow_is_trump {
        return Side::Lead;
    }
    if !lead_is_trump && follow_is_trump {
        return Side::Follow;
    }
    // Same trump-ness: both trump, both led-suit, or follower threw off.
    if follow.suit != led.suit {
        return Side::Lead;
    }
    if follow.rank > led.rank {
        Side::Follow
    } else {
        Side::Lead
    }
}

/// Lite per-round scoring (non-monotonic): 0-3 and 7-9 both score 6; 10-13 = 0.
pub fn score_for_tricks(n: u32) -> u32 {
    if n <= 3 {
        6
    } else if n == 4 {
        1
    } else if n == 5 {
        2
    } else if n == 6 {
        3
    } else if n <= 9 {
        6
    } else {
        0
    }
}

impl State {
    /// Deal a fresh round with the given starting score.
    pub fn deal_round<R: Rng + ?Sized>(round_num: u32, score: [u32; 2], rng: &mut R) -> State {
        let mut deck = create_deck();
        shuffle(&mut deck, rng);
        let mut human_hand: Vec<Card> = deck[0..13].to_vec();
        let mut bot_hand: Vec<Card> = deck[13..26].to_vec();
        let trump = deck[26];
        sort_hand(&mut human_hand);
        sort_hand(&mut bot_hand);
        let leader = initial_leader_for(round_num);
        State {
            human_hand,
            bot_hand,
            trump,
            leader,
            led_card: None,
            awaiting: Some(leader),
            tricks_won: [0, 0],
            score,
            round_num,
            trick_num: 1,
            phase: Phase::Playing,
            trick_history: Vec::new(),
        }
    }

    /// Start a new match (round 1, score 0-0).
    pub fn new_match<R: Rng + ?Sized>(rng: &mut R) -> State {
        State::deal_round(1, [0, 0], rng)
    }

    /// Construct a round directly from a recorded deal (for parity replay).
    pub fn from_round_setup(
        round_num: u32,
        score: [u32; 2],
        human_hand: Vec<Card>,
        bot_hand: Vec<Card>,
        trump: Card,
        leader: Player,
    ) -> State {
        State {
            human_hand,
            bot_hand,
            trump,
            leader,
            led_card: None,
            awaiting: Some(leader),
            tricks_won: [0, 0],
            score,
            round_num,
            trick_num: 1,
            phase: Phase::Playing,
            trick_history: Vec::new(),
        }
    }

    #[inline]
    fn hand_mut(&mut self, player: Player) -> &mut Vec<Card> {
        match player {
            Player::Human => &mut self.human_hand,
            Player::Bot => &mut self.bot_hand,
        }
    }

    #[inline]
    pub fn hand(&self, player: Player) -> &[Card] {
        match player {
            Player::Human => &self.human_hand,
            Player::Bot => &self.bot_hand,
        }
    }

    /// Legal moves for the player to move (`awaiting`).
    pub fn legal(&self) -> Vec<Card> {
        let p = self.awaiting.expect("legal() with no awaiting player");
        legal_moves(self.hand(p), self.led_card)
    }

    /// Apply a single card play by `awaiting`. Caller guarantees the card is in
    /// that player's hand and (for parity) legal.
    pub fn apply(&mut self, card: Card) {
        let player = self.awaiting.expect("apply() with no awaiting player");
        let hand = self.hand_mut(player);
        let pos = hand
            .iter()
            .position(|c| *c == card)
            .expect("apply(): card not in hand");
        hand.remove(pos);
        self.trick_history.push(PlayEvent {
            trick: self.trick_num,
            player,
            card,
        });

        match self.led_card {
            None => {
                // Leading the trick.
                self.led_card = Some(card);
                self.awaiting = Some(player.other());
            }
            Some(led) => {
                // Following — resolve the trick.
                let winner = match trick_winner(led, card, self.trump.suit) {
                    Side::Lead => self.leader,
                    Side::Follow => player,
                };
                self.tricks_won[winner.idx()] += 1;
                self.led_card = None;
                self.awaiting = None;
                self.leader = winner;
                self.phase = Phase::TrickComplete;
            }
        }
    }

    /// Move from a completed trick to the next (or to round-over).
    pub fn advance_after_trick(&mut self) {
        let next = self.trick_num + 1;
        self.trick_num = next;
        if next > TRICKS_PER_ROUND {
            self.awaiting = None;
            self.phase = Phase::RoundOver;
        } else {
            self.awaiting = Some(self.leader);
            self.phase = Phase::Playing;
        }
    }

    /// Apply round-end scoring; either deal the next round or end the match.
    pub fn end_round<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let human_pts = score_for_tricks(self.tricks_won[Player::Human.idx()]);
        let bot_pts = score_for_tricks(self.tricks_won[Player::Bot.idx()]);
        let new_score = [
            self.score[Player::Human.idx()] + human_pts,
            self.score[Player::Bot.idx()] + bot_pts,
        ];
        if new_score[0] >= TARGET_SCORE || new_score[1] >= TARGET_SCORE {
            self.score = new_score;
            self.awaiting = None;
            self.phase = Phase::MatchOver;
        } else {
            *self = State::deal_round(self.round_num + 1, new_score, rng);
        }
    }

    /// The match winner, once `phase == MatchOver`. Per the official rules, a tie
    /// on total points is broken in favor of whoever scored more in the final
    /// round (`tricks_won` still holds that round). Per-round point totals can
    /// never tie — the two sides split 13 tricks and `score_for_tricks` maps
    /// every such split to two different values — so this is always decisive.
    pub fn match_winner(&self) -> Option<Player> {
        if self.phase != Phase::MatchOver {
            return None;
        }
        let h = self.score[Player::Human.idx()];
        let b = self.score[Player::Bot.idx()];
        if h > b {
            Some(Player::Human)
        } else if b > h {
            Some(Player::Bot)
        } else {
            let hl = score_for_tricks(self.tricks_won[Player::Human.idx()]);
            let bl = score_for_tricks(self.tricks_won[Player::Bot.idx()]);
            if hl > bl {
                Some(Player::Human)
            } else {
                Some(Player::Bot)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_index_roundtrip() {
        for i in 0..NUM_CARDS {
            assert_eq!(Card::from_index(i).index(), i);
        }
        assert_eq!(Card::new(0, 1).index(), 0); // bells-1
        assert_eq!(Card::new(2, 11).index(), 32); // moons-11
    }

    #[test]
    fn scoring_table() {
        let expected = [6, 6, 6, 6, 1, 2, 3, 6, 6, 6, 0, 0, 0, 0];
        for (n, &pts) in expected.iter().enumerate() {
            assert_eq!(score_for_tricks(n as u32), pts, "tricks={n}");
        }
    }

    #[test]
    fn trick_winner_rules() {
        let trump = 2u8; // moons trump
                         // higher of led suit wins
        assert_eq!(
            trick_winner(Card::new(0, 5), Card::new(0, 9), trump),
            Side::Follow
        );
        assert_eq!(
            trick_winner(Card::new(0, 9), Card::new(0, 5), trump),
            Side::Lead
        );
        // throw-off (off-suit, non-trump) loses
        assert_eq!(
            trick_winner(Card::new(0, 2), Card::new(1, 11), trump),
            Side::Lead
        );
        // follower trumps
        assert_eq!(
            trick_winner(Card::new(0, 11), Card::new(2, 1), trump),
            Side::Follow
        );
        // leader trumps, follower off-suit
        assert_eq!(
            trick_winner(Card::new(2, 1), Card::new(0, 11), trump),
            Side::Lead
        );
        // both trump, higher wins
        assert_eq!(
            trick_winner(Card::new(2, 3), Card::new(2, 7), trump),
            Side::Follow
        );
    }

    #[test]
    fn must_follow_suit() {
        let hand = vec![Card::new(0, 3), Card::new(0, 8), Card::new(1, 5)];
        let legal = legal_moves(&hand, Some(Card::new(0, 1)));
        assert_eq!(legal, vec![Card::new(0, 3), Card::new(0, 8)]);
        // void in led suit -> anything
        let legal2 = legal_moves(&hand, Some(Card::new(2, 1)));
        assert_eq!(legal2.len(), 3);
        // leading -> anything
        assert_eq!(legal_moves(&hand, None).len(), 3);
    }

    #[test]
    fn match_winner_breaks_tie_by_final_round() {
        let mut rng = rand::thread_rng();
        let mut s = State::new_match(&mut rng);
        s.phase = Phase::MatchOver;

        // Higher total simply wins.
        s.score = [25, 20];
        s.tricks_won = [0, 13];
        assert_eq!(s.match_winner(), Some(Player::Human));

        // Tie on total -> whoever scored more in the final round wins.
        // Human took 7 tricks (6 pts) vs bot's 6 tricks (3 pts).
        s.score = [24, 24];
        s.tricks_won = [7, 6];
        assert_eq!(s.match_winner(), Some(Player::Human));

        // Mirror: human took 5 tricks (2 pts) vs bot's 8 tricks (6 pts).
        s.tricks_won = [5, 8];
        assert_eq!(s.match_winner(), Some(Player::Bot));
    }

    #[test]
    fn full_round_plays_13_tricks() {
        let mut rng = rand::thread_rng();
        let mut s = State::new_match(&mut rng);
        let mut tricks = 0;
        loop {
            match s.phase {
                Phase::Playing => {
                    let legal = s.legal();
                    s.apply(legal[0]);
                }
                Phase::TrickComplete => {
                    tricks += 1;
                    s.advance_after_trick();
                }
                Phase::RoundOver => break,
                Phase::MatchOver => break,
            }
        }
        assert_eq!(tricks, 13);
        assert_eq!(
            s.tricks_won[0] + s.tricks_won[1],
            13,
            "all tricks accounted for"
        );
        assert_eq!(s.human_hand.len() + s.bot_hand.len(), 0, "hands emptied");
    }
}
