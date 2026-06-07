//! Parity test: the Rust encoder must reproduce the reference JS encoder
//! (src/engine/encode.js) bit-for-bit at every recorded decision point.
//!
//! Regenerate the fixture:
//!   node training/scripts/dump_encode_fixtures.mjs 60 > training/fixtures/encode_fixtures.json

use foxlite_core::encode::{encode, legal_mask};
use foxlite_core::{Card, Phase, Player, PlayEvent, State};
use serde::Deserialize;

const SUITS: [&str; 3] = ["bells", "keys", "moons"];

fn suit_idx(s: &str) -> u8 {
    SUITS.iter().position(|x| *x == s).expect("suit") as u8
}
fn parse_player(p: &str) -> Player {
    match p {
        "human" => Player::Human,
        "bot" => Player::Bot,
        o => panic!("player {o}"),
    }
}

#[derive(Deserialize)]
struct JCard {
    suit: String,
    rank: u8,
}
impl JCard {
    fn to_card(&self) -> Card {
        Card::new(suit_idx(&self.suit), self.rank)
    }
}

#[derive(Deserialize)]
struct JScore {
    human: u32,
    bot: u32,
}

#[derive(Deserialize)]
struct JEvent {
    trick: u32,
    player: String,
    card: JCard,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct JState {
    human_hand: Vec<JCard>,
    bot_hand: Vec<JCard>,
    trump: JCard,
    led_card: Option<JCard>,
    tricks_won: JScore,
    score: JScore,
    round_num: u32,
    trick_num: u32,
    leader: String,
    awaiting: String,
    trick_history: Vec<JEvent>,
}

#[derive(Deserialize)]
struct Case {
    state: JState,
    mover: String,
    enc: Vec<usize>,
    mask: Vec<usize>,
}

#[derive(Deserialize)]
struct Fixtures {
    #[serde(rename = "inputSize")]
    input_size: usize,
    cases: Vec<Case>,
}

fn build_state(j: &JState) -> State {
    State {
        human_hand: j.human_hand.iter().map(|c| c.to_card()).collect(),
        bot_hand: j.bot_hand.iter().map(|c| c.to_card()).collect(),
        trump: j.trump.to_card(),
        leader: parse_player(&j.leader),
        led_card: j.led_card.as_ref().map(|c| c.to_card()),
        awaiting: Some(parse_player(&j.awaiting)),
        tricks_won: [j.tricks_won.human, j.tricks_won.bot],
        score: [j.score.human, j.score.bot],
        round_num: j.round_num,
        trick_num: j.trick_num,
        phase: Phase::Playing,
        trick_history: j
            .trick_history
            .iter()
            .map(|e| PlayEvent {
                trick: e.trick,
                player: parse_player(&e.player),
                card: e.card.to_card(),
            })
            .collect(),
    }
}

fn set_indices(v: &[f32]) -> Vec<usize> {
    v.iter()
        .enumerate()
        .filter(|(_, &x)| x != 0.0)
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn encoder_matches_js() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../fixtures/encode_fixtures.json"
    );
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {path}: {e}\n\
             generate: node training/scripts/dump_encode_fixtures.mjs 60 > training/fixtures/encode_fixtures.json"
        )
    });
    let fx: Fixtures = serde_json::from_str(&raw).expect("parse fixtures");
    assert_eq!(fx.input_size, foxlite_core::encode::INPUT_SIZE);
    assert!(!fx.cases.is_empty());

    for (i, case) in fx.cases.iter().enumerate() {
        let state = build_state(&case.state);
        let mover = parse_player(&case.mover);
        let enc = set_indices(&encode(&state, mover));
        assert_eq!(enc, case.enc, "case {i}: encoding mismatch");
        let mask = set_indices(&legal_mask(&state, mover));
        assert_eq!(mask, case.mask, "case {i}: legal mask mismatch");
    }
    eprintln!("encode parity OK over {} cases", fx.cases.len());
}
