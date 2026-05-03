<#
Push current repo to team GitHub repo (PowerShell)
Usage:
  - Set `GITHUB_TOKEN` env var (recommended) or use `gh auth login` beforehand.
  - Run in repo root:
      ./scripts/push_to_team_repo.ps1 -RemoteUrl 'https://github.com/MarBarb/sentiment-cross-domain.git'
#>
param(
    [string]
    $RemoteUrl = 'https://github.com/MarBarb/sentiment-cross-domain.git',
    [string]
    $RemoteName = 'team'
)

function Fail($msg){ Write-Error $msg; exit 1 }

# Check git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail 'git not found in PATH. Install Git and retry.' }

# Ensure inside git repo
$inside = git rev-parse --is-inside-work-tree 2>$null
if ($LASTEXITCODE -ne 0 -or $inside -ne 'true') { Fail 'Current directory is not inside a git repository. CD to the repo root and retry.' }

# Current branch
$branch = git symbolic-ref --short HEAD 2>$null
if ($LASTEXITCODE -ne 0 -or -not $branch) { $branch = 'main' }
Write-Host "Current branch: $branch"

# Use token if available
$token = $env:GITHUB_TOKEN
if ($token) {
    if ($RemoteUrl -match '^https://') {
        $authUrl = $RemoteUrl -replace '^https://', "https://$token@"
    } else {
        $authUrl = $RemoteUrl
    }
    Write-Host 'Using GITHUB_TOKEN from environment for HTTPS auth (token will not be printed)'
} else {
    $authUrl = $RemoteUrl
    Write-Host 'No GITHUB_TOKEN found. Ensure `gh auth login` or credential helper is configured.'
}

# Add or update remote
$existing = git remote | Select-String -Pattern "^$RemoteName$" -Quiet
if ($existing) {
    Write-Host "Remote '$RemoteName' exists — updating URL"
    git remote set-url $RemoteName $authUrl
} else {
    Write-Host "Adding remote '$RemoteName' -> $RemoteUrl"
    git remote add $RemoteName $authUrl
}

# Fetch and push
Write-Host 'Fetching remote refs...'
git fetch $RemoteName --prune
if ($LASTEXITCODE -ne 0) { Write-Warning 'Fetch failed; continuing to push may still work.' }

Write-Host "Pushing branch $branch to $RemoteName/$branch (will set upstream)..."
$pushCmd = "git push --set-upstream $RemoteName $branch"
Write-Host $pushCmd
& git push --set-upstream $RemoteName $branch
if ($LASTEXITCODE -ne 0) {
    Fail "Push failed. Check permissions and remote URL. If using token, ensure it has repo:status and repo permissions."
}

Write-Host 'Push succeeded.'
Write-Host "Repository available at: $RemoteUrl (if you have permission)."

Write-Host 'Note: If you used a token in the URL, remove it from your shell history and environment after use.'
