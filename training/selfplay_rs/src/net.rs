//! Fox-Lite net v2 (training/net.py) in tch-rs: transformer history encoder +
//! residual MLP trunk.
//!
//! Weights are loaded from fp32 safetensors keyed by state_dict FQN
//! (hist_embed, hist_self_embed, hist_pos, hist_layers.{i}.{ln1,q,k,v,o,ln2,
//! fc1,fc2}, hist_ln, stem, blocks.{i}.{ln1,fc1,ln2,fc2}, policy_ln/policy_fc,
//! value_ln/value_fc1/value_fc2). `forward` matches PyTorch's FoxNet exactly
//! (LayerNorm eps 1e-5, exact-erf GELU, additive (valid-1)*1e9 key mask,
//! masked mean-pool, tanh value head). `reload` copies fresh weights in place
//! for cheap hot-swap between cohorts.

use std::collections::HashMap;

use tch::{Device, Kind, Tensor};

use foxlite_core::encode::{HIST, HIST_TOKENS, STATIC_SIZE, TOKEN_FEATS};

const LN_EPS: f64 = 1e-5;
const MASK_NEG: f64 = 1.0e9;
/// Attention heads — not recoverable from weight shapes; must match net.py.
const N_HEADS: i64 = 4;

pub struct Net {
    dev: Device,
    kind: Kind,
    n_blocks: usize,
    n_hist_layers: usize,
    width: i64,
    d_model: i64,
    p: HashMap<String, Tensor>,
}

fn load_map(path: &str, dev: Device, kind: Kind) -> HashMap<String, Tensor> {
    Tensor::read_safetensors(path)
        .unwrap_or_else(|e| panic!("read_safetensors {path}: {e}"))
        .into_iter()
        .map(|(k, v)| (k, v.to_device(dev).to_kind(kind)))
        .collect()
}

impl Net {
    pub fn load(path: &str, dev: Device, kind: Kind) -> Net {
        let p = load_map(path, dev, kind);
        assert!(
            p.contains_key("hist_embed.weight"),
            "{path}: no hist_embed.weight — v1 (pre-history) weights are not supported"
        );
        let n_blocks = (0..)
            .take_while(|i| p.contains_key(&format!("blocks.{i}.fc1.weight")))
            .count();
        let n_hist_layers = (0..)
            .take_while(|i| p.contains_key(&format!("hist_layers.{i}.q.weight")))
            .count();
        let width = p
            .get("stem.weight")
            .unwrap_or_else(|| panic!("missing stem.weight"))
            .size()[0];
        let d_model = p.get("hist_embed.weight").unwrap().size()[1];
        assert_eq!(d_model % N_HEADS, 0, "d_model {d_model} not divisible by {N_HEADS} heads");
        Net {
            dev,
            kind,
            n_blocks,
            n_hist_layers,
            width,
            d_model,
            p,
        }
    }

    pub fn device(&self) -> Device {
        self.dev
    }

    fn g(&self, k: &str) -> &Tensor {
        self.p.get(k).unwrap_or_else(|| panic!("missing param {k}"))
    }

    fn linear(&self, x: &Tensor, pfx: &str) -> Tensor {
        x.linear(
            self.g(&format!("{pfx}.weight")),
            Some(self.g(&format!("{pfx}.bias"))),
        )
    }

    fn ln(&self, x: &Tensor, pfx: &str, width: i64) -> Tensor {
        x.layer_norm(
            [width],
            Some(self.g(&format!("{pfx}.weight"))),
            Some(self.g(&format!("{pfx}.bias"))),
            LN_EPS,
            true,
        )
    }

    fn embed(&self, key: &str, indices: &Tensor) -> Tensor {
        Tensor::embedding(self.g(key), indices, -1, false, false)
    }

    /// (policy_logits [B,33], value [B]) — matches FoxNet.forward.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let b = x.size()[0];
        let t = HIST_TOKENS as i64;
        let head_dim = self.d_model / N_HEADS;

        let tok = x
            .narrow(1, 0, HIST as i64)
            .reshape([b, t, TOKEN_FEATS as i64]);
        let statik = x.narrow(1, HIST as i64, STATIC_SIZE as i64);
        let card = tok.select(2, 0).to_kind(Kind::Int64);
        let self_bit = tok.select(2, 1).to_kind(Kind::Int64);
        let valid = tok.select(2, 2); // [B,T]

        let mut h = self.embed("hist_embed.weight", &card)
            + self.embed("hist_self_embed.weight", &self_bit)
            + self.g("hist_pos");
        let addmask = ((&valid - 1.0) * MASK_NEG).reshape([b, 1, 1, t]);
        for i in 0..self.n_hist_layers {
            let pfx = format!("hist_layers.{i}");
            let inp = h.shallow_clone();
            let hn = self.ln(&inp, &format!("{pfx}.ln1"), self.d_model);
            let q = self
                .linear(&hn, &format!("{pfx}.q"))
                .reshape([b, t, N_HEADS, head_dim])
                .transpose(1, 2);
            let k = self
                .linear(&hn, &format!("{pfx}.k"))
                .reshape([b, t, N_HEADS, head_dim])
                .transpose(1, 2);
            let v = self
                .linear(&hn, &format!("{pfx}.v"))
                .reshape([b, t, N_HEADS, head_dim])
                .transpose(1, 2);
            let att = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
            let att = (att + &addmask).softmax(-1, self.kind);
            let a = att.matmul(&v).transpose(1, 2).reshape([b, t, self.d_model]);
            let h1 = inp + self.linear(&a, &format!("{pfx}.o"));
            let f = self.ln(&h1, &format!("{pfx}.ln2"), self.d_model);
            let f = self.linear(&f, &format!("{pfx}.fc1")).gelu("none");
            let f = self.linear(&f, &format!("{pfx}.fc2"));
            h = h1 + f;
        }
        let h = self.ln(&h, "hist_ln", self.d_model);
        let vm = valid.unsqueeze(-1); // [B,T,1]
        let pooled = (h * &vm).sum_dim_intlist([1].as_slice(), false, self.kind)
            / vm.sum_dim_intlist([1].as_slice(), false, self.kind).clamp_min(1.0);

        let mut h = self.linear(&Tensor::cat(&[statik, pooled], 1), "stem");
        for i in 0..self.n_blocks {
            let inp = h.shallow_clone();
            let a = self.ln(&inp, &format!("blocks.{i}.ln1"), self.width).gelu("none");
            let a = self.linear(&a, &format!("blocks.{i}.fc1"));
            let b = self.ln(&a, &format!("blocks.{i}.ln2"), self.width).gelu("none");
            let b = self.linear(&b, &format!("blocks.{i}.fc2"));
            h = inp + b;
        }
        let policy = self.linear(&self.ln(&h, "policy_ln", self.width), "policy_fc");
        let v = self.ln(&h, "value_ln", self.width);
        let v = self.linear(&v, "value_fc1").gelu("none");
        let v = self.linear(&v, "value_fc2").tanh().squeeze_dim(-1);
        (policy, v)
    }

    /// Copy fresh weights in place (same keys/shapes); cheap between cohorts.
    pub fn reload(&self, path: &str) {
        let m = load_map(path, self.dev, self.kind);
        for (k, dst) in &self.p {
            if let Some(src) = m.get(k) {
                let mut t = dst.shallow_clone();
                let _ = t.copy_(src);
            }
        }
    }
}
