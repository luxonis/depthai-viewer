//! Helpers for error handling.

/// Format an error, including its chain of sources.
///
/// Always use this when displaying an error.
pub fn format(error: impl AsRef<dyn std::error::Error>) -> String {
    fn format_impl(error: &dyn std::error::Error) -> String {
        let mut string = error.to_string();
        for source in std::iter::successors(error.source(), |error| error.source()) {
            string.push_str(" -> ");
            string.push_str(&source.to_string());
        }
        string
    }

    format_impl(error.as_ref())
}

#[test]
fn test_format() {
    let err = anyhow::format_err!("root_cause")
        .context("inner_context")
        .context("outer_context");

    assert_eq!(err.to_string(), "outer_context"); // Oh no, we don't see the root cause!

    // Now we do:
    assert_eq!(format(&err), "outer_context -> inner_context -> root_cause");
}
