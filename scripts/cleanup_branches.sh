#!/usr/bin/env bash
#
# cleanup_branches.sh — NWC 分支治理：统计活跃分支、清理已合并分支与过期 worktree
#
# 关联 Issue #12，规则见 docs/engineering/BRANCH_POLICY.md。
#
# 统计口径：本地分支（refs/heads）+ origin 远程分支，同名去重，不含基准分支
# （默认 main）。远程清单以 `git ls-remote --heads origin` 为准，不依赖可能过期
# 的本地远程跟踪引用。默认上限 30；清理后仍超限时列出候选并以状态 1 退出，
# --force 可忽略该退出码。
#
# 默认 --dry-run：只统计并打印清理计划，不修改任何分支或 worktree；
# 只有 --execute 才真正删除。清理是幂等的：第二次执行不会重复删除。
#
# 依赖：git（>= 2.17，worktree + --porcelain -z）、gh（已登录；Actions 中通过
# GH_TOKEN 提供 GITHUB_TOKEN）、bash >= 4.4（关联数组）。仅支持 Linux/macOS 的
# 常见 bash 环境；Windows 请使用 WSL 或 Git Bash。
#
# shellcheck disable=SC2034  # LOCAL_SKIP / REMOTE_SKIP 由 print_skips 通过 nameref 读取
set -euo pipefail

PROG="$(basename "${BASH_SOURCE[0]}")"

LIMIT=30
BASE="main"
MODE="dry-run"          # dry-run | execute
FORCE=0
REPO=""

LOCAL_BRANCHES=""
REMOTE_BRANCHES=""
MERGED_LOCAL=""
PROTECTED=""
BASE_REF=""
CURRENT_BRANCH=""
CURRENT_WT=""
MAIN_WT=""

declare -A MERGED_PR_BRANCH=()   # 已合并 PR 的 head 分支 -> PR 编号
declare -A OPEN_PR_BRANCH=()     # 仍有 open PR 的 head 分支
declare -A LOCAL_SKIP=()         # 已合并但被跳过的本地分支 -> 原因
declare -A REMOTE_SKIP=()        # 已合并 PR 但被跳过的远程分支 -> 原因

declare -a LOCAL_DELETE=()       # 待删除的本地分支
declare -a REMOTE_DELETE=()      # 待删除的远程分支
declare -a WT_REMOVE=()          # 待移除的 worktree 路径

declare -a WT_PATHS=()           # 所有 worktree 路径（与 WT_BRANCHES 对应）
declare -a WT_BRANCHES=()        # worktree 检出的分支
declare -a WT_LOCKED=()          # worktree 是否 locked（1/0）

usage() {
  cat <<EOF
$PROG — NWC 分支治理脚本（Issue #12）

用法:
  bash scripts/$PROG [选项]

选项:
  --dry-run          只统计并打印清理计划，不删除任何东西（默认）
  --execute          执行删除：已合并的本地分支、已合并 PR 的远程分支、对应 worktree
  --force            清理后仍超过上限时忽略非零退出码（只影响退出码，不影响清理）
  --base <branch>    基准分支，默认 main
  --limit <N>        活跃分支上限（正整数），默认 30
  -h, --help         显示本帮助

统计口径:
  本地分支 + origin 远程分支，同名只算 1 个，不含基准分支，即"活跃分支"。

清理规则:
  本地      已合并进基准分支、不是基准/当前分支、未被其它 worktree 占用
  远程      已合并 PR 的 head 分支，且仍存在于 origin、没有 open PR、不受分支保护；
            跳过 fork（跨仓库）分支
  worktree  指向上述已合并分支、工作区干净、未 locked、不是当前/主 worktree；
            用 \`git worktree remove --force\` 移除，随后 \`git worktree prune\`

退出码:
  0  正常（dry-run 下表示"清理后不会超限"）
  1  清理后活跃分支仍超过上限（未加 --force），或删除操作失败
  2  参数/环境错误（不在仓库内、缺少 gh、读取远程/PR 失败等）

示例:
  bash scripts/cleanup_branches.sh --dry-run
  bash scripts/cleanup_branches.sh --execute
  bash scripts/cleanup_branches.sh --execute --force

注意:
  - 默认只读；在共享工作区里执行 --execute 前先看清"清理计划"。
  - squash 合并的分支不在 \`git branch --merged\` 结果中（提交不是 main 的祖先），
    由"已合并 PR"这条路径负责清理远程分支，本地同名分支需人工确认后删除。
EOF
}

log()  { printf '%s\n' "$*"; }
say()  { printf '\n== %s ==\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --dry-run) MODE="dry-run" ;;
      --execute) MODE="execute" ;;
      --force)   FORCE=1 ;;
      --base)
        [[ $# -ge 2 ]] || die "--base 需要一个分支名"
        BASE="$2"
        shift
        ;;
      --base=*) BASE="${1#--base=}" ;;
      --limit)
        [[ $# -ge 2 ]] || die "--limit 需要一个正整数"
        LIMIT="$2"
        shift
        ;;
      --limit=*) LIMIT="${1#--limit=}" ;;
      -h|--help) usage; exit 0 ;;
      *) die "未知参数: $1（用 --help 查看用法）" ;;
    esac
    shift
  done

  [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || die "--limit 必须是正整数，当前为: $LIMIT"
  [[ -n "$BASE" ]] || die "--base 不能为空"
  git check-ref-format --branch "$BASE" >/dev/null 2>&1 \
    || die "--base 不是合法的分支名: $BASE"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"
}

enter_repo() {
  local root
  root="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || die "当前目录不在 git 仓库中（请在仓库根目录或任意子目录运行）"
  cd "$root"
  git remote get-url origin >/dev/null 2>&1 \
    || die "仓库没有 origin 远程，无法统计/删除远程分支"
}

detect_repo() {
  if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
    REPO="$GITHUB_REPOSITORY"
  else
    REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null)" \
      || die "无法确定 GitHub 仓库（gh 未登录或 origin 不是 GitHub 仓库）"
  fi
  [[ -n "$REPO" ]] || die "无法确定 GitHub 仓库名"
}

count_lines() {
  if [[ -z "${1:-}" ]]; then
    printf '0\n'
  else
    printf '%s\n' "$1" | wc -l | tr -d ' '
  fi
}

load_local_branches() {
  LOCAL_BRANCHES="$(git for-each-ref --format='%(refname:short)' refs/heads \
    | sed '/^$/d' | sort -u)"
  CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'HEAD')"
  CURRENT_WT="$(git rev-parse --show-toplevel)"
}

load_remote_branches() {
  local out
  out="$(git ls-remote --heads origin)" \
    || die "无法读取 origin 远程分支（检查网络与凭据）"
  REMOTE_BRANCHES="$(printf '%s\n' "$out" \
    | awk '{print $2}' \
    | sed -e 's#^refs/heads/##' \
    | sed '/^$/d' | sort -u)"
}

resolve_base_ref() {
  if git rev-parse --verify --quiet "refs/remotes/origin/$BASE" >/dev/null; then
    BASE_REF="refs/remotes/origin/$BASE"
  elif git rev-parse --verify --quiet "refs/heads/$BASE" >/dev/null; then
    BASE_REF="refs/heads/$BASE"
  else
    die "找不到基准分支 $BASE（先执行 git fetch origin $BASE）"
  fi
}

load_merged_local() {
  MERGED_LOCAL="$(git for-each-ref --merged "$BASE_REF" \
    --format='%(refname:short)' refs/heads \
    | sed '/^$/d' | grep -vxF -- "$BASE" | sort -u || true)"
}

load_worktrees() {
  local -a lines=()
  local line path="" branch="" locked=0

  WT_PATHS=()
  WT_BRANCHES=()
  WT_LOCKED=()
  MAIN_WT=""

  mapfile -d '' -t lines < <(git worktree list --porcelain -z)

  for line in "${lines[@]}"; do
    case "$line" in
      "")
        # 空字段 = 一条 worktree 记录的结束
        if [[ -n "$path" && -z "$MAIN_WT" ]]; then
          MAIN_WT="$path"
        fi
        if [[ -n "$branch" ]]; then
          WT_PATHS+=("$path")
          WT_BRANCHES+=("$branch")
          WT_LOCKED+=("$locked")
        fi
        path=""
        branch=""
        locked=0
        ;;
      "worktree "*) path="${line#worktree }" ;;
      "branch "*)   branch="${line#branch refs/heads/}" ;;
      "locked"|"locked "*) locked=1 ;;
    esac
  done

  # 兼容结尾没有空字段的 git 版本：补一次 flush
  if [[ -n "$path" || -n "$branch" ]]; then
    if [[ -n "$path" && -z "$MAIN_WT" ]]; then
      MAIN_WT="$path"
    fi
    if [[ -n "$branch" ]]; then
      WT_PATHS+=("$path")
      WT_BRANCHES+=("$branch")
      WT_LOCKED+=("$locked")
    fi
  fi
}

load_pr_state() {
  local merged="" open="" num head cross
  MERGED_PR_BRANCH=()
  OPEN_PR_BRANCH=()

  if ! merged="$(gh pr list --repo "$REPO" --state merged --limit 500 \
        --json number,headRefName,isCrossRepository \
        --jq '.[] | "\(.number)\t\(.headRefName)\t\(.isCrossRepository)"' 2>/dev/null)"; then
    die "无法读取已合并 PR 列表（gh pr list 失败；检查 gh 登录、仓库名与网络）"
  fi
  while IFS=$'\t' read -r num head cross; do
    [[ -n "$head" ]] || continue
    [[ "$head" == "$BASE" ]] && continue
    [[ "$cross" == "true" ]] && continue
    # 同一分支可能有多个已合并 PR（重开/重复提交），保留最早的编号用于展示
    if [[ -z "${MERGED_PR_BRANCH[$head]:-}" ]]; then
      MERGED_PR_BRANCH["$head"]="$num"
    fi
  done <<< "$merged"

  if ! open="$(gh pr list --repo "$REPO" --state open --limit 200 \
        --json headRefName --jq '.[].headRefName' 2>/dev/null)"; then
    die "无法读取 open PR 列表（gh pr list 失败）"
  fi
  while IFS= read -r head; do
    [[ -n "$head" ]] && OPEN_PR_BRANCH["$head"]=1
  done <<< "$open"
}

load_protected() {
  local out=""
  if out="$(gh api --paginate "repos/$REPO/branches?protected=true&per_page=100" \
        --jq '.[].name' 2>/dev/null)"; then
    PROTECTED="$out"
  else
    warn "无法读取受保护分支列表，仅保护基准分支 $BASE"
    PROTECTED="$BASE"
  fi
  [[ -n "$PROTECTED" ]] || PROTECTED="$BASE"
}

is_protected() {
  local b="$1"
  [[ "$b" == "$BASE" ]] && return 0
  grep -qxF -- "$b" <<< "$PROTECTED"
}

has_remote_branch() {
  grep -qxF -- "$1" <<< "$REMOTE_BRANCHES"
}

list_active_branches() {
  {
    printf '%s\n' "$LOCAL_BRANCHES"
    printf '%s\n' "$REMOTE_BRANCHES"
  } | sed '/^$/d' \
    | grep -vxF -- "$BASE" \
    | sort -u || true
}

count_active() {
  list_active_branches | wc -l | tr -d ' '
}

same_path() {
  local a b
  a="$(realpath -m -- "$1" 2>/dev/null || printf '%s' "$1")"
  b="$(realpath -m -- "$2" 2>/dev/null || printf '%s' "$2")"
  [[ "$a" == "$b" ]]
}

worktrees_for_branch() {
  local b="$1" i
  for i in "${!WT_PATHS[@]}"; do
    if [[ "${WT_BRANCHES[$i]}" == "$b" ]]; then
      printf '%s\n' "${WT_PATHS[$i]}"
    fi
  done
  return 0
}

worktree_locked() {
  local p="$1" i
  for i in "${!WT_PATHS[@]}"; do
    if same_path "${WT_PATHS[$i]}" "$p"; then
      [[ "${WT_LOCKED[$i]}" == "1" ]]
      return $?
    fi
  done
  return 1
}

worktree_is_dirty() {
  local p="$1" out
  if ! out="$(git -C "$p" status --porcelain 2>/dev/null)"; then
    return 0   # 无法读取状态时按"脏"处理，宁可不删
  fi
  [[ -n "$out" ]]
}

plan_local() {
  local b p blocked
  local -a wt_paths=()

  LOCAL_DELETE=()
  LOCAL_SKIP=()

  while IFS= read -r b; do
    [[ -n "$b" ]] || continue
    if [[ "$b" == "$CURRENT_BRANCH" ]]; then
      LOCAL_SKIP["$b"]="当前分支"
      continue
    fi

    wt_paths=()
    mapfile -t wt_paths < <(worktrees_for_branch "$b")
    if (( ${#wt_paths[@]} > 0 )); then
      blocked=""
      for p in "${wt_paths[@]}"; do
        if same_path "$p" "$CURRENT_WT"; then
          blocked="当前 worktree: $p"
          break
        fi
        if same_path "$p" "$MAIN_WT"; then
          blocked="主 worktree: $p"
          break
        fi
        if worktree_locked "$p"; then
          blocked="worktree 已 locked: $p"
          break
        fi
        if worktree_is_dirty "$p"; then
          blocked="worktree 有未提交改动: $p"
          break
        fi
      done
      if [[ -n "$blocked" ]]; then
        LOCAL_SKIP["$b"]="$blocked"
        continue
      fi
      for p in "${wt_paths[@]}"; do
        WT_REMOVE+=("$p")
      done
    fi
    LOCAL_DELETE+=("$b")
  done <<< "$MERGED_LOCAL"
}

plan_remote() {
  local b
  REMOTE_DELETE=()
  REMOTE_SKIP=()

  while IFS= read -r b; do
    [[ -n "$b" ]] || continue
    if ! has_remote_branch "$b"; then
      REMOTE_SKIP["$b"]="远程已不存在"
      continue
    fi
    if [[ -n "${OPEN_PR_BRANCH[$b]:-}" ]]; then
      REMOTE_SKIP["$b"]="仍有 open PR"
      continue
    fi
    if is_protected "$b"; then
      REMOTE_SKIP["$b"]="受保护分支"
      continue
    fi
    REMOTE_DELETE+=("$b")
  done < <(printf '%s\n' "${!MERGED_PR_BRANCH[@]}" | sed '/^$/d' | sort)
}

print_skips() {
  local -n skip_map="$1"
  local label="$2"
  local b
  if (( ${#skip_map[@]} == 0 )); then
    return 0
  fi
  log "$label"
  while IFS= read -r b; do
    log "  - $b（${skip_map[$b]}）"
  done < <(printf '%s\n' "${!skip_map[@]}" | sed '/^$/d' | sort)
}

print_plan() {
  local p b
  say "清理计划（模式：$MODE）"

  if (( ${#WT_REMOVE[@]} == 0 && ${#LOCAL_DELETE[@]} == 0 && ${#REMOTE_DELETE[@]} == 0 )); then
    log "无可清理项。"
  else
    if (( ${#WT_REMOVE[@]} > 0 )); then
      log "worktree 移除：${#WT_REMOVE[@]} 个"
      for p in "${WT_REMOVE[@]}"; do
        log "  - $p"
      done
    fi
    if (( ${#LOCAL_DELETE[@]} > 0 )); then
      log "本地分支删除：${#LOCAL_DELETE[@]} 个"
      for b in "${LOCAL_DELETE[@]}"; do
        log "  - $b"
      done
    fi
    if (( ${#REMOTE_DELETE[@]} > 0 )); then
      log "远程分支删除：${#REMOTE_DELETE[@]} 个"
      for b in "${REMOTE_DELETE[@]}"; do
        log "  - origin/$b（已合并 PR #${MERGED_PR_BRANCH[$b]}）"
      done
    fi
  fi

  print_skips LOCAL_SKIP "跳过本地已合并分支："
  print_skips REMOTE_SKIP "跳过远程已合并 PR 分支："

  if [[ "$MODE" == "dry-run" ]]; then
    log "（dry-run：以上仅为计划，未删除任何东西；加 --execute 才执行）"
  fi
}

execute_plan() {
  local p b rc=0

  if (( ${#WT_REMOVE[@]} == 0 && ${#LOCAL_DELETE[@]} == 0 && ${#REMOTE_DELETE[@]} == 0 )); then
    log "没有需要执行的清理。"
    return 0
  fi

  say "执行清理"

  for p in "${WT_REMOVE[@]}"; do
    log "移除 worktree: $p"
    if ! git worktree remove --force -- "$p"; then
      warn "worktree 移除失败，保留登记: $p"
      rc=1
    fi
  done
  if (( ${#WT_REMOVE[@]} > 0 )); then
    git worktree prune >/dev/null 2>&1 || warn "git worktree prune 失败（忽略）"
  fi

  for b in "${LOCAL_DELETE[@]}"; do
    if git show-ref --verify --quiet "refs/heads/$b"; then
      if git branch -D -- "$b"; then
        log "已删除本地分支: $b"
      else
        warn "本地分支删除失败（可能仍被 worktree 占用）: $b"
        rc=1
      fi
    else
      log "本地分支已不存在，跳过: $b"
    fi
  done

  for b in "${REMOTE_DELETE[@]}"; do
    log "删除远程分支: origin/$b"
    if git push --quiet origin --delete -- "$b"; then
      log "已删除远程分支: origin/$b"
    else
      warn "远程分支删除失败: origin/$b"
      rc=1
    fi
  done

  return "$rc"
}

print_branch_table() {
  local b tags date
  while IFS= read -r b; do
    [[ -n "$b" ]] || continue
    tags=""
    grep -qxF -- "$b" <<< "$LOCAL_BRANCHES" && tags+="local "
    grep -qxF -- "$b" <<< "$REMOTE_BRANCHES" && tags+="remote "
    if [[ -n "${OPEN_PR_BRANCH[$b]:-}" ]]; then
      tags+="open-pr "
    fi
    if [[ -n "${MERGED_PR_BRANCH[$b]:-}" ]]; then
      tags+="merged-pr#${MERGED_PR_BRANCH[$b]} "
    fi
    grep -qxF -- "$b" <<< "$MERGED_LOCAL" && tags+="merged-local "
    date=""
    if git show-ref --verify --quiet "refs/heads/$b"; then
      date="$(git log -1 --format='%cs' "refs/heads/$b" 2>/dev/null || true)"
    fi
    printf '  - %-42s %s%s\n' "$b" "$tags" "${date:+（最后提交 $date）}"
  done <<< "$(list_active_branches)"
}

main() {
  parse_args "$@"
  require_cmd git
  require_cmd gh
  enter_repo
  detect_repo
  resolve_base_ref

  load_local_branches
  load_remote_branches
  load_worktrees
  load_pr_state
  load_protected
  load_merged_local

  local before after local_count remote_count rc=0
  before="$(count_active)"

  plan_local
  plan_remote

  say "统计（上限 $LIMIT；不含 $BASE）"
  local_count="$(count_lines "$LOCAL_BRANCHES")"
  remote_count="$(count_lines "$REMOTE_BRANCHES")"
  log "本地分支：$local_count"
  log "远程分支（origin）：$remote_count"
  log "活跃分支（去重）：$before"

  print_plan

  if [[ "$MODE" == "execute" ]]; then
    if ! execute_plan; then
      rc=1
    fi
    # 重新读取，让"清理后"计数反映真实结果
    load_local_branches
    load_remote_branches
  fi

  local_count="$(count_lines "$LOCAL_BRANCHES")"
  remote_count="$(count_lines "$REMOTE_BRANCHES")"
  after="$(count_active)"

  say "结果"
  log "清理前：活跃分支 $before"
  log "清理后：本地 $local_count / 远程 $remote_count / 活跃（去重） $after / 上限 $LIMIT"

  if (( rc != 0 )); then
    warn "部分删除操作失败，请按上面的告警人工处理。"
  fi

  if (( after > LIMIT )); then
    say "超过上限：清理后仍有 $after 个活跃分支（上限 $LIMIT）"
    log "当前活跃分支："
    print_branch_table
    log "处理建议：合并/关闭长期 PR；删除已合并分支；回收不用的 worktree。超限期间暂停新建分支。"
    if (( FORCE == 1 )); then
      warn "--force：忽略超限退出码"
    else
      log "以非零状态退出（加 --force 可忽略；详见 --help）。"
      exit 1
    fi
  fi

  if (( rc != 0 )); then
    exit 1
  fi
  exit 0
}

main "$@"
