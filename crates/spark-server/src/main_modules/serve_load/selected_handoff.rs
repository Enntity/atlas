// SPDX-License-Identifier: AGPL-3.0-only

//! Retain the selected owner while sharing read-only startup metadata queries.
use spark_model::traits::Model;

pub(super) enum ServingModel {
    Ordinary(Box<dyn Model>),
    #[cfg(target_os = "linux")]
    Selected(crate::glm_terminal_session::SelectedModel),
}

impl std::ops::Deref for ServingModel {
    type Target = dyn Model;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ordinary(model) => model.as_ref(),
            #[cfg(target_os = "linux")]
            Self::Selected(owner) => owner.model(),
        }
    }
}
