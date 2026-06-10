//! selfplay_rs CLI.
//!
//! Subcommands:
//!   forward-check <dir>   Phase-3 parity gate: Rust tch forward vs PyTorch fixture
//!                         (fwd_weights.safetensors + fwd_fixture.safetensors in <dir>)

use std::collections::HashMap;
use std::time::Duration;

use tch::{Device, Kind, Tensor};

use selfplay_rs::aoti::AotiModel;
use selfplay_rs::net::Net;
use selfplay_rs::pipeline;
use selfplay_rs::pipeline::{ENC_LEN, POLICY_SIZE};

/// Read a `--key value` flag, falling back to `default`.
fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// Shared CLI parsing for the `serve` and `bench` self-play subcommands.
fn parse_pipeline_config(args: &[String]) -> pipeline::Config {
    pipeline::Config {
        sims: flag(args, "--sims", "400").parse().unwrap(),
        add_root_noise: !args.iter().any(|a| a == "--no-noise"),
        seed: flag(args, "--seed", "0").parse().unwrap(),
        n_threads: flag(args, "--threads", "16").parse().unwrap(),
        n_slots: flag(args, "--slots", "2").parse().unwrap(),
        batch: flag(args, "--batch", "512").parse().unwrap(),
        weights_path: flag(args, "--weights", "serving_weights.safetensors"),
        model_path: flag(args, "--model", ""),
        reload_every: Duration::from_millis(flag(args, "--reload-ms", "2000").parse().unwrap()),
        cpu: args.iter().any(|a| a == "--cpu"),
    }
}

fn read_fixture(path: &str) -> HashMap<String, Tensor> {
    Tensor::read_safetensors(path)
        .unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
        .into_iter()
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
    let a = a.to_kind(Kind::Float).to_device(Device::Cpu);
    let b = b.to_kind(Kind::Float).to_device(Device::Cpu);
    (a - b).abs().max().double_value(&[])
}

fn forward_check(dir: &str) -> bool {
    let wpath = format!("{dir}/fwd_weights.safetensors");
    let fix = read_fixture(&format!("{dir}/fwd_fixture.safetensors"));
    let input = fix.get("input").expect("fixture.input");
    let ref_logits = fix.get("ref_logits").expect("fixture.ref_logits");
    let ref_value = fix.get("ref_value").expect("fixture.ref_value");
    let n = input.size()[0];

    // ---- CPU fp32: exact-math gate vs PyTorch CPU fp32 (no TF32 noise) ----
    let net_cpu = Net::load(&wpath, Device::Cpu, Kind::Float);
    let x_cpu = input.to_device(Device::Cpu).to_kind(Kind::Float);
    let (pl, vl) = net_cpu.forward(&x_cpu);
    let dl = max_abs_diff(&pl, ref_logits);
    let dv = max_abs_diff(&vl, ref_value);
    println!("forward-check on {n} positions:");
    println!("  CPU fp32 vs PyTorch: max|Δlogits|={dl:.3e}  max|Δvalue|={dv:.3e}");
    let cpu_ok = dl < 1e-4 && dv < 1e-4;

    // ---- GPU smoke: prove CUDA path runs and is close (fp32 + bf16) ----
    let mut gpu_ok = true;
    if tch::Cuda::is_available() {
        let dev = Device::Cuda(0);
        let net_g = Net::load(&wpath, dev, Kind::Float);
        let xg = input.to_device(dev).to_kind(Kind::Float);
        let (plg, vlg) = net_g.forward(&xg);
        let dlg = max_abs_diff(&plg, ref_logits);
        let dvg = max_abs_diff(&vlg, ref_value);

        let net_b = Net::load(&wpath, dev, Kind::BFloat16);
        let xb = input.to_device(dev).to_kind(Kind::BFloat16);
        let (plb, vlb) = net_b.forward(&xb);
        let dlb = max_abs_diff(&plb, ref_logits);
        let dvb = max_abs_diff(&vlb, ref_value);
        println!("  GPU fp32 vs PyTorch:  max|Δlogits|={dlg:.3e}  max|Δvalue|={dvg:.3e}");
        println!("  GPU bf16 vs PyTorch:  max|Δlogits|={dlb:.3e}  max|Δvalue|={dvb:.3e}");
        // fp32 GPU may use TF32 (loose); bf16 looser still — sanity bounds only.
        gpu_ok = dlg < 5e-2 && dvg < 5e-2 && dlb < 2e-1;
    } else {
        println!("  (CUDA unavailable — skipping GPU smoke)");
    }

    if !cpu_ok {
        println!("  FAIL: CPU fp32 diverges from PyTorch (math/layout bug)");
    }
    if !gpu_ok {
        println!("  FAIL: GPU forward outside sanity bounds");
    }
    cpu_ok && gpu_ok
}

/// Parity gate for the AOTInductor path: compare the fused `.pt2` forward (run via
/// the C++ shim) against the eager tch bf16 forward on the same random input. Both
/// are bf16, so this isolates fusion/accumulation-order noise from the C++ link/run
/// path being correct. Returns true if they agree within the bf16 sanity bound.
fn aoti_check(weights: &str, model: &str, batch: i64) -> bool {
    if !tch::Cuda::is_available() {
        println!("aoti-check: CUDA unavailable — skipping");
        return true;
    }
    let dev = Device::Cuda(0);
    let x = Tensor::randn([batch, ENC_LEN as i64], (Kind::Float, dev)).to_kind(Kind::BFloat16);

    // Eager bf16 reference.
    let net = Net::load(weights, dev, Kind::BFloat16);
    let (ref_l, ref_v) = net.forward(&x);

    // Fused AOTI forward via the shim.
    let aoti = AotiModel::load(model);
    match aoti.check_batch(batch, ENC_LEN as i64) {
        0 => {}
        code => {
            println!("aoti-check: FAIL — {model} not compiled for batch={batch} (check_batch={code})");
            return false;
        }
    }
    let out_l = Tensor::zeros([batch, POLICY_SIZE as i64], (Kind::BFloat16, dev));
    let out_v = Tensor::zeros([batch], (Kind::BFloat16, dev));
    aoti.run(&x, &out_l, &out_v);

    let dl = max_abs_diff(&out_l, &ref_l);
    let dv = max_abs_diff(&out_v, &ref_v);
    println!("aoti-check on batch={batch}:");
    println!("  AOTI vs eager bf16:  max|Δlogits|={dl:.3e}  max|Δvalue|={dv:.3e}");
    // Both paths are bf16; only fusion/accumulation order differs. Loose sanity bound.
    let ok = dl < 2e-1 && dv < 1e-1;
    println!("{}", if ok { "AOTI-CHECK OK" } else { "AOTI-CHECK FAILED" });
    ok
}

/// Verify the in-place weight hot-swap path: load a `.pt2` (built from some other
/// weights), push the weights from `w2` via `swap_weights`, and confirm the fused
/// forward now matches the EAGER forward of `w2`. If the mangled constant names did
/// not match, the swap would silently no-op and AOTI would NOT match net(w2) — so
/// this catches a broken hot-reload (which would otherwise train on stale weights).
fn aoti_swap_check(model: &str, w2: &str, batch: i64) -> bool {
    if !tch::Cuda::is_available() {
        println!("aoti-swap-check: CUDA unavailable — skipping");
        return true;
    }
    let dev = Device::Cuda(0);
    let aoti = AotiModel::load(model);
    if aoti.check_batch(batch, ENC_LEN as i64) != 0 {
        println!("aoti-swap-check: FAIL — {model} not compiled for batch={batch}");
        return false;
    }
    // Push w2's weights into the loaded package (mangled FQN '.'->'_', bf16 CUDA).
    let entries = Tensor::read_safetensors(w2).expect("read w2 safetensors");
    let mut names: Vec<std::ffi::CString> = Vec::with_capacity(entries.len());
    let mut tensors: Vec<Tensor> = Vec::with_capacity(entries.len());
    for (name, t) in entries {
        names.push(std::ffi::CString::new(name.replace('.', "_")).unwrap());
        tensors.push(t.to_device(dev).to_kind(Kind::BFloat16));
    }
    let refs: Vec<&Tensor> = tensors.iter().collect();
    aoti.swap_weights(&names, &refs);

    let x = Tensor::randn([batch, ENC_LEN as i64], (Kind::Float, dev)).to_kind(Kind::BFloat16);
    let out_l = Tensor::zeros([batch, POLICY_SIZE as i64], (Kind::BFloat16, dev));
    let out_v = Tensor::zeros([batch], (Kind::BFloat16, dev));
    aoti.run(&x, &out_l, &out_v);

    let (ref_l, ref_v) = Net::load(w2, dev, Kind::BFloat16).forward(&x);
    let dl = max_abs_diff(&out_l, &ref_l);
    let dv = max_abs_diff(&out_v, &ref_v);
    println!("aoti-swap-check on batch={batch} (swapped in {w2}):");
    println!("  AOTI(after swap) vs eager bf16(w2):  max|Δlogits|={dl:.3e}  max|Δvalue|={dv:.3e}");
    let ok = dl < 2e-1 && dv < 1e-1;
    println!("{}", if ok { "AOTI-SWAP-CHECK OK" } else { "AOTI-SWAP-CHECK FAILED (stale weights!)" });
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    match cmd {
        "aoti-swap-check" => {
            let model = flag(&args, "--model", "serving_model.pt2");
            let w2 = flag(&args, "--w2", "serving_weights.safetensors");
            let batch: i64 = flag(&args, "--batch", "2048").parse().unwrap();
            std::process::exit(if aoti_swap_check(&model, &w2, batch) { 0 } else { 1 });
        }
        "aoti-check" => {
            let weights = flag(&args, "--weights", "serving_weights.safetensors");
            let model = flag(&args, "--model", "serving_model.pt2");
            let batch: i64 = flag(&args, "--batch", "2048").parse().unwrap();
            std::process::exit(if aoti_check(&weights, &model, batch) { 0 } else { 1 });
        }
        "forward-check" => {
            let dir = args.get(2).map(String::as_str).unwrap_or(".");
            let ok = forward_check(dir);
            println!("{}", if ok { "FORWARD-CHECK OK" } else { "FORWARD-CHECK FAILED" });
            std::process::exit(if ok { 0 } else { 1 });
        }
        "serve" => {
            // Continuous ISMCTS self-play worker: streams finished games as framed
            // bytes on stdout for the orchestrator; shuts down on stdin EOF.
            pipeline::run_serve(parse_pipeline_config(&args));
        }
        "bench" => {
            // Throughput probe (no trainer): prints games/sec + rows/sec.
            let run = flag(&args, "--run-secs", "60").parse().unwrap();
            let interval = flag(&args, "--interval-secs", "10").parse().unwrap();
            let warmup = flag(&args, "--warmup-secs", "90").parse().unwrap();
            pipeline::run_bench(
                parse_pipeline_config(&args),
                Duration::from_secs_f64(run),
                Duration::from_secs_f64(interval),
                Duration::from_secs_f64(warmup),
            );
        }
        other => {
            eprintln!("unknown subcommand {other:?}; expected: forward-check | serve | bench");
            std::process::exit(2);
        }
    }
}
