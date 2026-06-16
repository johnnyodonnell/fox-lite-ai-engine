//! Fox-Lite residual MLP (training/net.py) in tch-rs.
//!
//! Weights are loaded from fp32 safetensors keyed by state_dict FQN
//! (stem, blocks.{i}.{ln1,fc1,ln2,fc2}, policy_ln/policy_fc,
//! value_ln/value_fc1/value_fc2). `forward` matches PyTorch's FoxNet exactly
//! (LayerNorm eps 1e-5, exact-erf GELU, tanh value head). `reload` copies fresh
//! weights in place for cheap hot-swap between cohorts.

use std::collections::HashMap;

use tch::{Device, Kind, Tensor};

const LN_EPS: f64 = 1e-5;

pub struct Net {
    dev: Device,
    kind: Kind,
    n_blocks: usize,
    width: i64,
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
        let n_blocks = (0..)
            .take_while(|i| p.contains_key(&format!("blocks.{i}.fc1.weight")))
            .count();
        let width = p
            .get("stem.weight")
            .unwrap_or_else(|| panic!("missing stem.weight"))
            .size()[0];
        Net {
            dev,
            kind,
            n_blocks,
            width,
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

    fn ln(&self, x: &Tensor, pfx: &str) -> Tensor {
        x.layer_norm(
            [self.width],
            Some(self.g(&format!("{pfx}.weight"))),
            Some(self.g(&format!("{pfx}.bias"))),
            LN_EPS,
            true,
        )
    }

    /// (policy_logits [B,33], value [B]) — matches FoxNet.forward.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut h = self.linear(x, "stem");
        for i in 0..self.n_blocks {
            let inp = h.shallow_clone();
            let a = self.ln(&inp, &format!("blocks.{i}.ln1")).gelu("none");
            let a = self.linear(&a, &format!("blocks.{i}.fc1"));
            let b = self.ln(&a, &format!("blocks.{i}.ln2")).gelu("none");
            let b = self.linear(&b, &format!("blocks.{i}.fc2"));
            h = inp + b;
        }
        let policy = self.linear(&self.ln(&h, "policy_ln"), "policy_fc");
        let v = self.ln(&h, "value_ln");
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
