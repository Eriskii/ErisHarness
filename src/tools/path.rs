//! Path resolution as Pi does it: `@` prefixes are stripped, Unicode spaces become plain
//! spaces, `~` is the sandbox home, and relative paths resolve against the working directory.

use std::path::{Component, Path, PathBuf};

fn is_unicode_space(c: char) -> bool {
    matches!(c, '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}')
}

pub fn resolve(input: &str, cwd: &str, home: &str) -> String {
    let spaced: String = input.chars().map(|c| if is_unicode_space(c) { ' ' } else { c }).collect();
    let path = spaced.strip_prefix('@').unwrap_or(&spaced);
    let path = path.strip_prefix("file://").unwrap_or(path);
    let expanded = if path == "~" {
        home.to_owned()
    } else if let Some(rest) = path.strip_prefix("~/") {
        format!("{home}/{rest}")
    } else {
        path.to_owned()
    };
    let joined = Path::new(cwd).join(expanded);
    let mut normal = PathBuf::from("/");
    for component in joined.components() {
        match component {
            Component::Normal(part) => normal.push(part),
            Component::ParentDir => {
                normal.pop();
            }
            Component::RootDir | Component::CurDir | Component::Prefix(_) => {}
        }
    }
    normal.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::resolve;

    #[test]
    fn resolves_like_node_path_resolve() {
        assert_eq!(resolve("a/b", "/w", "/root"), "/w/a/b");
        assert_eq!(resolve("/x/../y/./z", "/w", "/root"), "/y/z");
        assert_eq!(resolve("../../..", "/w/a", "/root"), "/");
        assert_eq!(resolve("~", "/w", "/root"), "/root");
        assert_eq!(resolve("~/f", "/w", "/root"), "/root/f");
        assert_eq!(resolve("@src/x", "/w", "/root"), "/w/src/x");
        assert_eq!(resolve("a\u{00A0}b", "/w", "/root"), "/w/a b");
        assert_eq!(resolve("file:///etc/x", "/w", "/root"), "/etc/x");
    }
}
