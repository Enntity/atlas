// SPDX-License-Identifier: AGPL-3.0-only

use super::core::TerminalCore;

pub(super) fn install(core: &'static TerminalCore) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        core.panic_if_live();
        previous(info);
    }));
}
