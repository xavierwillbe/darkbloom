#!/bin/zsh
set -euo pipefail

export LC_ALL=C
export LANG=en_US.UTF-8

INSTALL_URL="${INSTALL_URL:-https://api.darkbloom.dev/install.sh}"
MODEL_ID="${MODEL_ID:-CohereLabs/cohere-transcribe-03-2026}"
COORDINATOR_WS="${COORDINATOR_WS:-wss://api.darkbloom.dev/ws/provider}"
PROFILE_URL="${PROFILE_URL:-https://api.darkbloom.dev/enroll.mobileconfig}"
DARKBLOOM_HOME="${DARKBLOOM_HOME:-$HOME/.darkbloom}"
DARKBLOOM_BIN="$DARKBLOOM_HOME/bin/darkbloom"
PATCH_WINDOW="${PATCH_WINDOW:-90}"
PATCH_INTERVAL="${PATCH_INTERVAL:-2}"
BACKEND_PORT="${BACKEND_PORT:-8100}"
MODE="${1:-earn}"

log() {
  printf '[darkbloom-fleet] %s\n' "$*"
}

fail() {
  printf '[darkbloom-fleet] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

ensure_darkbloom_installed() {
  if [[ -x "$DARKBLOOM_BIN" ]]; then
    log "Darkbloom already installed at $DARKBLOOM_BIN"
    return
  fi

  log "Installing Darkbloom"
  curl -fsSL "$INSTALL_URL" | env LC_ALL=C LANG=en_US.UTF-8 bash
}

ensure_path() {
  local line='export PATH="$HOME/.darkbloom/bin:$PATH"'
  if [[ -f "$HOME/.zshrc" ]] && grep -Fq "$line" "$HOME/.zshrc"; then
    return
  fi

  log "Adding Darkbloom to ~/.zshrc"
  {
    printf '\n# Darkbloom\n'
    printf '%s\n' "$line"
  } >> "$HOME/.zshrc"
}

ensure_linked() {
  if "$DARKBLOOM_BIN" status 2>/dev/null | grep -Fq 'Linked:   ✓ Yes'; then
    log "Account already linked"
    return
  fi

  log "Account not linked yet; starting interactive login"
  "$DARKBLOOM_BIN" login

  if ! "$DARKBLOOM_BIN" status 2>/dev/null | grep -Fq 'Linked:   ✓ Yes'; then
    fail "login did not complete; rerun after approving the device"
  fi
}

ensure_model() {
  log "Ensuring model is installed: $MODEL_ID"
  "$DARKBLOOM_BIN" install \
    --coordinator "$COORDINATOR_WS" \
    --profile-url "$PROFILE_URL" \
    --model "$MODEL_ID" \
    || true
}

patch_tokenizer() {
  python3 <<'PYEOF'
from pathlib import Path

path = Path.home() / ".darkbloom/python/lib/python3.12/site-packages/mlx_audio/stt/models/cohere_asr/tokenizer.py"
path.parent.mkdir(parents=True, exist_ok=True)
text = """import json
from pathlib import Path
from typing import Iterable, List, Optional


class _TokenizersBackend:
    def __init__(self, tokenizer_json_path: str):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(tokenizer_json_path)

    def piece_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return -1 if token_id is None else int(token_id)

    def id_to_piece(self, token_id: int) -> str:
        token = self.tokenizer.id_to_token(int(token_id))
        return "" if token is None else token

    def get_piece_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text).ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(
            [int(token_id) for token_id in ids],
            skip_special_tokens=False,
        )


class _SentencePieceBackend:
    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def piece_to_id(self, token: str) -> int:
        return int(self.sp.piece_to_id(token))

    def id_to_piece(self, token_id: int) -> str:
        return self.sp.id_to_piece(int(token_id))

    def get_piece_size(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str) -> List[int]:
        return list(self.sp.encode(text))

    def decode(self, ids: Iterable[int]) -> str:
        return self.sp.decode([int(token_id) for token_id in ids])


class CohereAsrTokenizer:
    def __init__(
        self,
        model_path: str,
        tokenizer_config_path: Optional[str] = None,
        special_tokens_map_path: Optional[str] = None,
    ):
        resolved_model_path = Path(model_path)
        tokenizer_json_path = self._resolve_tokenizer_json_path(resolved_model_path)

        if resolved_model_path.exists() and resolved_model_path.suffix != ".json":
            self.sp = _SentencePieceBackend(str(resolved_model_path))
        elif tokenizer_json_path is not None and tokenizer_json_path.exists():
            self.sp = _TokenizersBackend(str(tokenizer_json_path))
        else:
            raise FileNotFoundError(
                f"Could not locate tokenizer.model or tokenizer.json near {model_path}"
            )

        tokenizer_config = self._load_json(tokenizer_config_path)
        special_tokens_map = self._load_json(special_tokens_map_path)

        self.bos_token = tokenizer_config.get(
            "bos_token",
            special_tokens_map.get("bos_token", "<|startoftranscript|>"),
        )
        self.eos_token = tokenizer_config.get(
            "eos_token",
            special_tokens_map.get("eos_token", "<|endoftext|>"),
        )
        self.pad_token = tokenizer_config.get(
            "pad_token",
            special_tokens_map.get("pad_token", "<pad>"),
        )
        self.unk_token = tokenizer_config.get(
            "unk_token",
            special_tokens_map.get("unk_token", "<unk>"),
        )

        additional = tokenizer_config.get("additional_special_tokens", [])
        if not additional:
            additional = special_tokens_map.get("additional_special_tokens", [])
        self.additional_special_tokens = list(additional)

        self.special_tokens = {
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.unk_token,
            *self.additional_special_tokens,
        }
        self.special_token_ids = {
            self.sp.piece_to_id(token)
            for token in self.special_tokens
            if self.sp.piece_to_id(token) >= 0
        }
        self.vocab_size = self.sp.get_piece_size()

    @staticmethod
    def _resolve_tokenizer_json_path(model_path: Path) -> Optional[Path]:
        candidates = []
        if model_path.suffix == ".json":
            candidates.append(model_path)
        candidates.append(model_path.with_name("tokenizer.json"))
        if model_path.parent != model_path:
            candidates.append(model_path.parent / "tokenizer.json")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_json(path: Optional[str]) -> dict:
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    @property
    def bos_token_id(self) -> int:
        return self.sp.piece_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.sp.piece_to_id(self.eos_token)

    @property
    def pad_token_id(self) -> int:
        return self.sp.piece_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        return self.sp.piece_to_id(self.unk_token)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        token_ids = list(self.sp.encode(text))
        if add_special_tokens:
            token_ids = [self.bos_token_id, *token_ids, self.eos_token_id]
        return token_ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        token_ids = [int(token_id) for token_id in ids if int(token_id) >= 0]
        if skip_special_tokens:
            filtered = [
                token_id for token_id in token_ids
                if token_id not in self.special_token_ids
            ]
            return self.sp.decode(filtered)

        output = []
        buffer = []
        for token_id in token_ids:
            piece = self.sp.id_to_piece(token_id)
            if piece in self.special_tokens:
                if buffer:
                    output.append(self.sp.decode(buffer))
                    buffer = []
                output.append(piece)
            else:
                buffer.append(token_id)
        if buffer:
            output.append(self.sp.decode(buffer))
        return "".join(output)

    def batch_decode(
        self, batch_ids: List[Iterable[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_ids
        ]

    def build_prompt_tokens(self, language: str, punctuation: bool = True) -> List[int]:
        tokens = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            f"<|{language}|>",
            f"<|{language}|>",
            "<|pnc|>" if punctuation else "<|nopnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        return [self.sp.piece_to_id(token) for token in tokens]
"""
path.write_text(text)
print(path)
PYEOF
}

write_launcher() {
  local mode="$1"
  local launcher="$DARKBLOOM_HOME/bin/darkbloom-$mode"
  local serve_flags

  if [[ "$mode" == "earn" ]]; then
    serve_flags='--model "$MODEL_ID" --backend-port "$BACKEND_PORT" --no-auto-update'
  else
    serve_flags='--local --model "$MODEL_ID" --backend-port "$BACKEND_PORT" --no-auto-update'
  fi

  cat > "$launcher" <<EOF
#!/bin/zsh
set -euo pipefail

export LC_ALL=C
export LANG=en_US.UTF-8

DARKBLOOM_HOME="\$HOME/.darkbloom"
DARKBLOOM_BIN="\$DARKBLOOM_HOME/bin/darkbloom"
MODEL_ID="${MODEL_ID}"
BACKEND_PORT="${BACKEND_PORT}"
PATCH_WINDOW="${PATCH_WINDOW}"
PATCH_INTERVAL="${PATCH_INTERVAL}"

apply_patch() {
  python3 <<'PYEOF'
from pathlib import Path

path = Path.home() / ".darkbloom/python/lib/python3.12/site-packages/mlx_audio/stt/models/cohere_asr/tokenizer.py"
path.parent.mkdir(parents=True, exist_ok=True)
text = """import json
from pathlib import Path
from typing import Iterable, List, Optional


class _TokenizersBackend:
    def __init__(self, tokenizer_json_path: str):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(tokenizer_json_path)

    def piece_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return -1 if token_id is None else int(token_id)

    def id_to_piece(self, token_id: int) -> str:
        token = self.tokenizer.id_to_token(int(token_id))
        return "" if token is None else token

    def get_piece_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text).ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(
            [int(token_id) for token_id in ids],
            skip_special_tokens=False,
        )


class _SentencePieceBackend:
    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def piece_to_id(self, token: str) -> int:
        return int(self.sp.piece_to_id(token))

    def id_to_piece(self, token_id: int) -> str:
        return self.sp.id_to_piece(int(token_id))

    def get_piece_size(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str) -> List[int]:
        return list(self.sp.encode(text))

    def decode(self, ids: Iterable[int]) -> str:
        return self.sp.decode([int(token_id) for token_id in ids])


class CohereAsrTokenizer:
    def __init__(
        self,
        model_path: str,
        tokenizer_config_path: Optional[str] = None,
        special_tokens_map_path: Optional[str] = None,
    ):
        resolved_model_path = Path(model_path)
        tokenizer_json_path = self._resolve_tokenizer_json_path(resolved_model_path)

        if resolved_model_path.exists() and resolved_model_path.suffix != ".json":
            self.sp = _SentencePieceBackend(str(resolved_model_path))
        elif tokenizer_json_path is not None and tokenizer_json_path.exists():
            self.sp = _TokenizersBackend(str(tokenizer_json_path))
        else:
            raise FileNotFoundError(
                f"Could not locate tokenizer.model or tokenizer.json near {model_path}"
            )

        tokenizer_config = self._load_json(tokenizer_config_path)
        special_tokens_map = self._load_json(special_tokens_map_path)

        self.bos_token = tokenizer_config.get(
            "bos_token",
            special_tokens_map.get("bos_token", "<|startoftranscript|>"),
        )
        self.eos_token = tokenizer_config.get(
            "eos_token",
            special_tokens_map.get("eos_token", "<|endoftext|>"),
        )
        self.pad_token = tokenizer_config.get(
            "pad_token",
            special_tokens_map.get("pad_token", "<pad>"),
        )
        self.unk_token = tokenizer_config.get(
            "unk_token",
            special_tokens_map.get("unk_token", "<unk>"),
        )

        additional = tokenizer_config.get("additional_special_tokens", [])
        if not additional:
            additional = special_tokens_map.get("additional_special_tokens", [])
        self.additional_special_tokens = list(additional)

        self.special_tokens = {
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.unk_token,
            *self.additional_special_tokens,
        }
        self.special_token_ids = {
            self.sp.piece_to_id(token)
            for token in self.special_tokens
            if self.sp.piece_to_id(token) >= 0
        }
        self.vocab_size = self.sp.get_piece_size()

    @staticmethod
    def _resolve_tokenizer_json_path(model_path: Path) -> Optional[Path]:
        candidates = []
        if model_path.suffix == ".json":
            candidates.append(model_path)
        candidates.append(model_path.with_name("tokenizer.json"))
        if model_path.parent != model_path:
            candidates.append(model_path.parent / "tokenizer.json")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_json(path: Optional[str]) -> dict:
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    @property
    def bos_token_id(self) -> int:
        return self.sp.piece_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.sp.piece_to_id(self.eos_token)

    @property
    def pad_token_id(self) -> int:
        return self.sp.piece_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        return self.sp.piece_to_id(self.unk_token)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        token_ids = list(self.sp.encode(text))
        if add_special_tokens:
            token_ids = [self.bos_token_id, *token_ids, self.eos_token_id]
        return token_ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        token_ids = [int(token_id) for token_id in ids if int(token_id) >= 0]
        if skip_special_tokens:
            filtered = [
                token_id for token_id in token_ids
                if token_id not in self.special_token_ids
            ]
            return self.sp.decode(filtered)

        output = []
        buffer = []
        for token_id in token_ids:
            piece = self.sp.id_to_piece(token_id)
            if piece in self.special_tokens:
                if buffer:
                    output.append(self.sp.decode(buffer))
                    buffer = []
                output.append(piece)
            else:
                buffer.append(token_id)
        if buffer:
            output.append(self.sp.decode(buffer))
        return "".join(output)

    def batch_decode(
        self, batch_ids: List[Iterable[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_ids
        ]

    def build_prompt_tokens(self, language: str, punctuation: bool = True) -> List[int]:
        tokens = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            f"<|{language}|>",
            f"<|{language}|>",
            "<|pnc|>" if punctuation else "<|nopnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        return [self.sp.piece_to_id(token) for token in tokens]
"""
path.write_text(text)
print(path)
PYEOF
}

port_open() {
  python3 - "\$1" <<'PYEOF'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    s.close()
PYEOF
}

cleanup() {
  if [[ -n "\${PATCHER_PID:-}" ]]; then
    kill "\$PATCHER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

apply_patch
"\$DARKBLOOM_BIN" stop >/dev/null 2>&1 || true
if command -v lsof >/dev/null 2>&1; then
  existing_pids=(\$(lsof -tiTCP:"\$BACKEND_PORT" -sTCP:LISTEN 2>/dev/null || true))
  if (( \${#existing_pids[@]} > 0 )); then
    kill \${existing_pids[@]} >/dev/null 2>&1 || true
    sleep 1
  fi
fi

(
  end=\$((SECONDS + PATCH_WINDOW))
  while (( SECONDS < end )); do
    apply_patch >/dev/null 2>&1 || true
    if port_open "\$BACKEND_PORT"; then
      break
    fi
    sleep "\$PATCH_INTERVAL"
  done
) &
PATCHER_PID=\$!

exec "\$DARKBLOOM_BIN" serve ${serve_flags} "\$@"
EOF

  chmod +x "$launcher"
  log "Wrote launcher: $launcher"
}

main() {
  require_cmd curl
  require_cmd python3

  ensure_darkbloom_installed
  ensure_path
  ensure_linked
  ensure_model
  patch_tokenizer
  write_launcher stt
  write_launcher earn

  if [[ "$MODE" == "earn" ]]; then
    log "Starting earning mode"
    exec "$DARKBLOOM_HOME/bin/darkbloom-earn"
  fi

  if [[ "$MODE" == "stt" ]]; then
    log "Starting local STT mode"
    exec "$DARKBLOOM_HOME/bin/darkbloom-stt"
  fi

  log "Setup complete"
  log "Run darkbloom-earn for coordinator mode, or darkbloom-stt for local mode"
}

main "$@"
