#!/usr/bin/env bash
# postAttachCommand: install Claude Code plugins into the container's ~/.claude, which is lost
# on rebuild. Idempotent. The claude binary ships inside the VS Code extension, which may not
# be installed yet when this starts, hence the wait loop.
set -uo pipefail

find_claude_bin() {
    find "$HOME/.vscode-server/extensions" -maxdepth 1 -iname 'anthropic.claude-code-*' -type d 2> /dev/null \
        | sort -V | tail -1 | sed 's#$#/resources/native-binary/claude#'
}

CLAUDE_BIN=""
for _ in $(seq 1 30); do
    candidate="$(find_claude_bin)"
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        CLAUDE_BIN="$candidate"
        break
    fi
    sleep 2
done

if [ -z "$CLAUDE_BIN" ]; then
    echo "setup-claude-plugins: Claude Code binary not found; skipping plugin setup." >&2
    exit 0
fi

# Already installed is reported as an error by the CLI
"$CLAUDE_BIN" plugin marketplace add obra/superpowers > /dev/null 2>&1 || true
"$CLAUDE_BIN" plugin install superpowers@superpowers-dev > /dev/null 2>&1 || true

echo "setup-claude-plugins: superpowers plugin installed."
