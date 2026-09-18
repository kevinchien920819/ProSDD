#!/usr/bin/env bash
# Extract 4-second / 200-frame VAD prosody targets for ASVspoof datasets.
#
# Usage:
#   bash extract.sh <vad_checkpoint_dir> [19la] [21la] [21df] [spoof5]
#
# With no dataset selector, extract every split below.  Selectors are
# case-insensitive; for example, `bash extract.sh /path/to/checkpoint 21LA`.
# Outputs are written to prosody_vad_txt/ and replace files with the same name.

set -euo pipefail

usage() {
  cat <<'EOF'
用法：bash extract.sh <vad_checkpoint_dir> [19la] [21la] [21df] [spoof5]

未指定資料集時，依序抽取：
  19la   ASVspoof 2019 LA：train、dev、eval
  21la   ASVspoof 2021 LA：eval
  21df   ASVspoof 2021 DF：eval
  spoof5 ASVspoof5 Track 1：train、dev、eval

<vad_checkpoint_dir> 必須包含 model_config.yml 與 pytorch_model.bin。
ASVspoof2021 的 trial_metadata.txt 只會保留第 8 欄為 eval 的資料列。
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  if [[ $# -eq 0 ]]; then
    exit 1
  fi
  exit 0
fi

checkpoint_dir="$1"
shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ ! -d "$checkpoint_dir" ]]; then
  echo "錯誤：VAD checkpoint 目錄不存在：$checkpoint_dir" >&2
  exit 1
fi
if [[ ! -f "$checkpoint_dir/model_config.yml" || ! -f "$checkpoint_dir/pytorch_model.bin" ]]; then
  echo "錯誤：VAD checkpoint 必須包含 model_config.yml 與 pytorch_model.bin：$checkpoint_dir" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv，請先依 README 安裝專案環境。" >&2
  exit 1
fi

mkdir -p prosody_vad_txt

declare -A wanted=(
  [19la]=0
  [21la]=0
  [21df]=0
  [spoof5]=0
)

if [[ $# -eq 0 ]]; then
  for dataset in "${!wanted[@]}"; do
    wanted["$dataset"]=1
  done
else
  for selector in "$@"; do
    selector="${selector,,}"
    if [[ "$selector" == "all" ]]; then
      for dataset in "${!wanted[@]}"; do
        wanted["$dataset"]=1
      done
    elif [[ -v "wanted[$selector]" ]]; then
      wanted["$selector"]=1
    else
      echo "錯誤：未知的資料集選項：$selector" >&2
      usage >&2
      exit 1
    fi
  done
fi

run_extract() {
  local name="$1"
  local protocol_txt="$2"
  local audio_dir="$3"
  local out_txt="$4"
  local utt_col="$5"

  if [[ ! -f "$protocol_txt" ]]; then
    echo "錯誤：$name 的 protocol 不存在：$protocol_txt" >&2
    exit 1
  fi
  if [[ ! -d "$audio_dir" ]]; then
    echo "錯誤：$name 的音檔目錄不存在：$audio_dir" >&2
    exit 1
  fi

  echo "==> 抽取 $name"
  uv run --locked python extract_Prosody_vad.py \
    --protocol_txt "$protocol_txt" \
    --audio_dir "$audio_dir" \
    --out_txt "$out_txt" \
    --utt_col "$utt_col" --ext .flac --layer 7 \
    --checkpoint_dir "$checkpoint_dir"
}

run_2021_eval_extract() {
  local name="$1"
  local metadata_txt="$2"
  local audio_dir="$3"
  local out_txt="$4"
  local temp_protocol

  if [[ ! -f "$metadata_txt" ]]; then
    echo "錯誤：$name 的 trial metadata 不存在：$metadata_txt" >&2
    exit 1
  fi

  # LA 與 DF 的 utterance ID 都在第 2 欄；第 8 欄是 subset。
  temp_protocol="$(mktemp "${TMPDIR:-/tmp}/prosdd-${name// /_}.XXXXXX")"
  awk '$8 == "eval"' "$metadata_txt" > "$temp_protocol"
  if [[ ! -s "$temp_protocol" ]]; then
    rm -f "$temp_protocol"
    echo "錯誤：$name 的 trial metadata 找不到第 8 欄為 eval 的資料列：$metadata_txt" >&2
    exit 1
  fi

  run_extract "$name (eval)" "$temp_protocol" "$audio_dir" "$out_txt" 1
  rm -f "$temp_protocol"
}

if [[ ${wanted[19la]} -eq 1 ]]; then
  run_extract "ASVspoof2019 LA train" \
    dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
    dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
    prosody_vad_txt/asvspoof2019_train_prosody.txt 1
  run_extract "ASVspoof2019 LA dev" \
    dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
    dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
    prosody_vad_txt/asvspoof2019_dev_prosody.txt 1
  run_extract "ASVspoof2019 LA eval" \
    dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
    dataset/ASVspoof2019/ASVspoof2019_LA_eval/flac \
    prosody_vad_txt/asvspoof2019_eval_prosody.txt 1
fi

if [[ ${wanted[21la]} -eq 1 ]]; then
  run_2021_eval_extract "ASVspoof2021 LA" \
    dataset/ASVspoof2021/keys/LA/CM/trial_metadata.txt \
    dataset/ASVspoof2021/ASVspoof2021_LA_eval/flac \
    prosody_vad_txt/asvspoof2021_la_eval_prosody.txt
fi

if [[ ${wanted[21df]} -eq 1 ]]; then
  run_2021_eval_extract "ASVspoof2021 DF" \
    dataset/ASVspoof2021/keys/DF/CM/trial_metadata.txt \
    dataset/ASVspoof2021/ASVspoof2021_DF_eval/flac \
    prosody_vad_txt/asvspoof2021_df_eval_prosody.txt
fi

if [[ ${wanted[spoof5]} -eq 1 ]]; then
  run_extract "ASVspoof5 train" \
    dataset/ASVspoof5/ASVspoof5.train.tsv \
    dataset/ASVspoof5/flac_T \
    prosody_vad_txt/asvspoof5_train_prosody.txt 1
  run_extract "ASVspoof5 dev" \
    dataset/ASVspoof5/ASVspoof5.dev.track_1.tsv \
    dataset/ASVspoof5/flac_D \
    prosody_vad_txt/asvspoof5_dev_prosody.txt 1
  run_extract "ASVspoof5 eval" \
    dataset/ASVspoof5/ASVspoof5.eval.track_1.tsv \
    dataset/ASVspoof5/flac_E_eval \
    prosody_vad_txt/asvspoof5_eval_prosody.txt 1
fi

echo "完成。Prosody targets 已寫入：$script_dir/prosody_vad_txt"
