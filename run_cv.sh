#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="config.toml"
FOLDS=5
START_FOLD=0
END_FOLD=""
PYTHON_BIN="python"
KEEP_CONFIG=0
declare -a OVERRIDES=()

usage() {
  cat <<'EOF'
Usage:
  ./run_cv.sh [options]

Options:
  -c, --config <path>        TOML config path (default: config.toml)
  -k, --folds <int>          Number of CV folds (default: 5)
  --start-fold <int>         Start fold index (default: 0)
  --end-fold <int>           End fold index, inclusive (default: folds-1)
  -p, --python <bin>         Python executable (default: python)
  -s, --set <key=value>      Override TOML key before every run (repeatable)
  --keep-config              Keep modified config after script exits
  -h, --help                 Show help

Examples:
  ./run_cv.sh
  ./run_cv.sh -k 5 --start-fold 0 --end-fold 4
  ./run_cv.sh -s dataset=cybhi -s epochs=30 -s baseline=true
EOF
}

set_toml_key() {
  local config="$1"
  local key="$2"
  local raw_value="$3"
  local py_bin="$4"

  "$py_bin" - "$config" "$key" "$raw_value" <<'PY'
import pathlib
import re
import sys

config_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
raw_value = sys.argv[3]

if not config_path.is_file():
    raise SystemExit(f"Config not found: {config_path}")

text = config_path.read_text(encoding="utf-8")

# If caller passed TOML literal (quoted string/bool/number/array/object/datetime), keep it.
# Otherwise, treat as string and quote it.
literal_pattern = re.compile(
    r"""^(
        "([^"\\]|\\.)*" |
        '([^'\\]|\\.)*' |
        true|false |
        [+-]?\d+(\.\d+)?([eE][+-]?\d+)? |
        \[.*\] |
        \{.*\} |
        \d{4}-\d{2}-\d{2}([Tt ].*)?
    )$""",
    re.VERBOSE,
)

if literal_pattern.match(raw_value):
    value = raw_value
else:
    value = '"' + raw_value.replace("\\", "\\\\").replace('"', '\\"') + '"'

pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$")
if not pattern.search(text):
    raise SystemExit(f"Key not found in {config_path}: {key}")

updated = pattern.sub(lambda m: f"{m.group(1)}{value}", text, count=1)
config_path.write_text(updated, encoding="utf-8")
PY
}

get_toml_key() {
  local config="$1"
  local key="$2"
  local py_bin="$3"

  "$py_bin" - "$config" "$key" <<'PY'
import pathlib
import sys

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python <=3.10

config_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]

if not config_path.is_file():
    raise SystemExit(f"Config not found: {config_path}")

data = tomllib.loads(config_path.read_text(encoding="utf-8"))

def find_key(obj, target):
    if isinstance(obj, dict):
        if target in obj and not isinstance(obj[target], dict):
            return obj[target]
        for value in obj.values():
            found = find_key(value, target)
            if found is not None:
                return found
    return None

value = find_key(data, key)
if value is None:
    raise SystemExit(f"Key not found in {config_path}: {key}")

if isinstance(value, bool):
    print(str(value).lower())
else:
    print(value)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -k|--folds)
      FOLDS="$2"
      shift 2
      ;;
    --start-fold)
      START_FOLD="$2"
      shift 2
      ;;
    --end-fold)
      END_FOLD="$2"
      shift 2
      ;;
    -p|--python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -s|--set)
      OVERRIDES+=("$2")
      shift 2
      ;;
    --keep-config)
      KEEP_CONFIG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

if ! [[ "$FOLDS" =~ ^[0-9]+$ ]] || (( FOLDS <= 1 )); then
  echo "--folds must be an integer > 1, got: $FOLDS" >&2
  exit 1
fi

if ! [[ "$START_FOLD" =~ ^[0-9]+$ ]]; then
  echo "--start-fold must be a non-negative integer, got: $START_FOLD" >&2
  exit 1
fi

if [[ -z "$END_FOLD" ]]; then
  END_FOLD=$((FOLDS - 1))
fi

if ! [[ "$END_FOLD" =~ ^[0-9]+$ ]]; then
  echo "--end-fold must be a non-negative integer, got: $END_FOLD" >&2
  exit 1
fi

if (( START_FOLD > END_FOLD )); then
  echo "start-fold ($START_FOLD) cannot be greater than end-fold ($END_FOLD)" >&2
  exit 1
fi

if (( END_FOLD >= FOLDS )); then
  echo "end-fold ($END_FOLD) must be < folds ($FOLDS)" >&2
  exit 1
fi

ORIG_CONFIG="$(mktemp)"
cp "$CONFIG_PATH" "$ORIG_CONFIG"

cleanup() {
  if (( KEEP_CONFIG == 0 )); then
    cp "$ORIG_CONFIG" "$CONFIG_PATH"
    echo "Restored original config: $CONFIG_PATH"
  else
    echo "Kept modified config: $CONFIG_PATH"
  fi
  rm -f "$ORIG_CONFIG"
}
trap cleanup EXIT

echo "Using config: $CONFIG_PATH"
echo "Running folds: ${START_FOLD}..${END_FOLD} (total folds setting: $FOLDS)"

for override in "${OVERRIDES[@]}"; do
  if [[ "$override" != *=* ]]; then
    echo "Invalid --set value: $override (expected key=value)" >&2
    exit 1
  fi
  key="${override%%=*}"
  value="${override#*=}"
  set_toml_key "$CONFIG_PATH" "$key" "$value" "$PYTHON_BIN"
done

KD_ENABLED="$(get_toml_key "$CONFIG_PATH" "kd" "$PYTHON_BIN")"
BASELINE_ENABLED="$(get_toml_key "$CONFIG_PATH" "baseline" "$PYTHON_BIN")"
DATASET="$(get_toml_key "$CONFIG_PATH" "dataset" "$PYTHON_BIN")"
TEACHER_MODEL="$(get_toml_key "$CONFIG_PATH" "teacher_model" "$PYTHON_BIN")"
TEACHER_DIR="models_para/${DATASET}/resnet34"

if [[ "$KD_ENABLED" == "true" && "$BASELINE_ENABLED" != "true" ]]; then
  if [[ ! -d "$TEACHER_DIR" ]]; then
    echo "Teacher checkpoint directory not found: $TEACHER_DIR" >&2
    exit 1
  fi
  echo "KD mode enabled. Teacher checkpoint dir: $TEACHER_DIR"
fi

for (( fold=START_FOLD; fold<=END_FOLD; fold++ )); do
  fold_human=$((fold + 1))
  echo "========== Fold ${fold_human}/${FOLDS} (cv_fold_idx=${fold}) =========="
  set_toml_key "$CONFIG_PATH" "cv_fold_idx" "$fold" "$PYTHON_BIN"
  if [[ "$KD_ENABLED" == "true" && "$BASELINE_ENABLED" != "true" ]]; then
    teacher_ckpt="${TEACHER_DIR}/${TEACHER_MODEL}_${DATASET}_f${fold_human}_kd.pth"
    if [[ ! -f "$teacher_ckpt" ]]; then
      echo "Teacher checkpoint not found for fold ${fold_human}: $teacher_ckpt" >&2
      exit 1
    fi
    echo "Teacher checkpoint: $teacher_ckpt"
    TEACHER_CKPT_PATH="$teacher_ckpt" "$PYTHON_BIN" main.py
  else
    "$PYTHON_BIN" main.py
  fi
done

echo "All folds finished."
