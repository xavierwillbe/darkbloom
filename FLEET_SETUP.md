# Darkbloom Fleet Setup

This repo includes scripts to bootstrap independent Mac minis into the repaired Darkbloom STT provider flow.

## Files

- `darkbloom_fleet_setup.sh`: installs Darkbloom if needed, ensures PATH, prompts for login when needed, installs the STT model, patches the tokenizer, and writes `darkbloom-earn` / `darkbloom-stt`
- `bootstrap_from_repo.sh`: clones or updates this repo on a target Mac and runs `darkbloom_fleet_setup.sh`

## Quick Start On A New Mac

```bash
curl -fsSL https://raw.githubusercontent.com/xavierwillbe/darkbloom/main/bootstrap_from_repo.sh -o /tmp/bootstrap_from_repo.sh
chmod +x /tmp/bootstrap_from_repo.sh
REPO_URL=https://github.com/xavierwillbe/darkbloom.git /tmp/bootstrap_from_repo.sh earn
```

Use `stt` instead of `earn` if you want local-only mode.

## SSH Rollout Example

```bash
for host in macmini001 macmini002 macmini003; do
  ssh "$host" 'curl -fsSL https://raw.githubusercontent.com/xavierwillbe/darkbloom/main/bootstrap_from_repo.sh -o /tmp/bootstrap_from_repo.sh && chmod +x /tmp/bootstrap_from_repo.sh && REPO_URL=https://github.com/xavierwillbe/darkbloom.git /tmp/bootstrap_from_repo.sh earn'
done
```

## Notes

- each machine still needs to complete `darkbloom login` unless it already has a linked account
- the script defaults to `CohereLabs/cohere-transcribe-03-2026`
- `darkbloom-earn` is the coordinator-connected mode
- `darkbloom-stt` is the local-only mode
