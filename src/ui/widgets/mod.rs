//! Reusable egui widgets that are independent of the editor's document
//! model. Each widget here must only depend on `egui` (and `serde` for
//! persistable view state) so it can later be extracted into its own
//! crate.

pub mod node_graph;
