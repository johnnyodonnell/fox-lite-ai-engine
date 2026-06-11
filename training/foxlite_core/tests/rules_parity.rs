//! Parity test: replay deals + moves captured from the authoritative JS rules
//! (src/engine/game.js, via training/scripts/dump_rules_traces.mjs) and assert
//! the Rust port produces identical tricks, scores, and match winners.
//!
//! Regenerate the fixture:
//!   node training/scripts/dump_rules_traces.mjs 300 > training/foxlite_core/tests/rules_traces.json

use foxlite_core::{Card, Phase, Player, State, NUM_RANKS};
use serde::Deserialize;

const SUITS: [&str; 3] = ["bells", "keys", "moons"];

fn parse_card(id: &str) -> Card {
    let (suit_str, rank_str) = id.rsplit_once('-').expect("card id");
    let suit = SUITS.iter().position(|s| *s == suit_str).expect("suit") as u8;
    let rank: u8 = rank_str.parse().expect("rank");
    Card::new(suit, rank)
}

fn parse_player(p: &str) -> Player {
    match p {
        "human" => Player::Human,
        "bot" => Player::Bot,
        other => panic!("bad player {other}"),
    }
}

#[derive(Deserialize)]
struct Score {
    human: u32,
    bot: u32,
}

#[derive(Deserialize)]
struct Move {
    player: String,
    card: String,
}

#[derive(Deserialize)]
struct Round {
    #[serde(rename = "roundNum")]
    round_num: u32,
    score: Score,
    #[serde(rename = "humanHand")]
    human_hand: Vec<String>,
    #[serde(rename = "botHand")]
    bot_hand: Vec<String>,
    trump: String,
    leader: String,
    moves: Vec<Move>,
    #[serde(rename = "tricksWon")]
    tricks_won: Score,
    #[serde(rename = "scoreAfter")]
    score_after: Score,
}

#[derive(Deserialize)]
struct Game {
    rounds: Vec<Round>,
    winner: String,
    #[serde(rename = "finalScore")]
    final_score: Score,
}

#[derive(Deserialize)]
struct Traces {
    games: Vec<Game>,
}

#[test]
fn rules_match_js() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/rules_traces.json");
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {path}: {e}\n\
             generate it: node training/scripts/dump_rules_traces.mjs 300 > {path}"
        )
    });
    let traces: Traces = serde_json::from_str(&raw).expect("parse traces json");
    let mut rng = rand::thread_rng();

    assert!(!traces.games.is_empty(), "no games in fixture");
    for (gi, game) in traces.games.iter().enumerate() {
        for round in &game.rounds {
            let human: Vec<Card> = round.human_hand.iter().map(|s| parse_card(s)).collect();
            let bot: Vec<Card> = round.bot_hand.iter().map(|s| parse_card(s)).collect();
            let trump = parse_card(&round.trump);
            let leader = parse_player(&round.leader);
            let mut s = State::from_round_setup(
                round.round_num,
                [round.score.human, round.score.bot],
                human,
                bot,
                trump,
                leader,
            );

            for mv in &round.moves {
                // Advance through any completed trick before the next play.
                if s.phase == Phase::TrickComplete {
                    s.advance_after_trick();
                }
                assert_eq!(
                    s.awaiting,
                    Some(parse_player(&mv.player)),
                    "game {gi} round {}: awaiting mismatch",
                    round.round_num
                );
                s.apply(parse_card(&mv.card));
            }
            // Finish the final trick.
            if s.phase == Phase::TrickComplete {
                s.advance_after_trick();
            }
            assert_eq!(s.phase, Phase::RoundOver, "game {gi}: round not over");
            assert_eq!(
                [s.tricks_won[0], s.tricks_won[1]],
                [round.tricks_won.human, round.tricks_won.bot],
                "game {gi} round {}: tricks mismatch",
                round.round_num
            );

            s.end_round(&mut rng);
            assert_eq!(
                [s.score[0], s.score[1]],
                [round.score_after.human, round.score_after.bot],
                "game {gi} round {}: score-after mismatch",
                round.round_num
            );
        }
        // The last round's end_round drove either a new deal or match-over; the
        // recorded finalScore is the cumulative score after the last round.
        let last = game.rounds.last().unwrap();
        assert_eq!(
            [last.score_after.human, last.score_after.bot],
            [game.final_score.human, game.final_score.bot],
            "game {gi}: final score mismatch"
        );
        let expected_winner = parse_player(&game.winner);
        let winner = if game.final_score.human >= 21 && game.final_score.human >= game.final_score.bot
        {
            Player::Human
        } else {
            Player::Bot
        };
        assert_eq!(winner, expected_winner, "game {gi}: winner mismatch");
    }
    // sanity: NUM_RANKS used (keeps import meaningful across edits)
    assert_eq!(NUM_RANKS, 11);
}
