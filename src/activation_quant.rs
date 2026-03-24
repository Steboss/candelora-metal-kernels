//! Phase-1 activation-quantization contract surface.
//!
//! This module intentionally introduces type/API contracts without changing
//! kernel math. Runtime integration can start using these types now while the
//! underlying W8A8/W4A8 kernels are implemented in later phases.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivationQuantMode {
    /// Keep existing floating-point activation path.
    Off,
    /// Contract target for W8A8 (INT8 activation) kernels.
    W8A8,
    /// Contract target for W4A8 workflows.
    W4A8,
}

impl Default for ActivationQuantMode {
    fn default() -> Self {
        Self::Off
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationQuantConfig {
    /// Activation quantization mode.
    pub mode: ActivationQuantMode,
    /// Strict mode lets callers reject unsupported/non-calibrated paths.
    pub strict: bool,
}

impl Default for ActivationQuantConfig {
    fn default() -> Self {
        Self {
            mode: ActivationQuantMode::Off,
            strict: false,
        }
    }
}

impl ActivationQuantConfig {
    pub fn is_enabled(&self) -> bool {
        !matches!(self.mode, ActivationQuantMode::Off)
    }
}
