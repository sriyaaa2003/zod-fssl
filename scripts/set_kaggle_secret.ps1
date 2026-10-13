# Creates the Modal secret 'kaggle-credentials' that lets a Modal job upload your private Kaggle dataset.
# Run it AFTER scripts\install_kaggle_token.ps1. Nothing is printed.
#   - the access token is read from ~\.kaggle\access_token (uploading files needs the newer token, not the old key)
#   - the username is read from ~\.kaggle\kaggle.json if present, otherwise it asks
#   cd C:\Users\sriya\zod-fssl
#   powershell -ExecutionPolicy Bypass -File .\scripts\set_kaggle_secret.ps1

$dir = Join-Path $HOME ".kaggle"
$legacy = Join-Path $dir "kaggle.json"
# the token may be saved as access_token or access_token.txt (Notepad adds .txt)
$token = $null; $tokenFile = $null
foreach ($name in @("access_token", "access_token.txt")) {
    $f = Join-Path $dir $name
    if (-not (Test-Path $f)) { continue }
    try {
        # remove spaces, line breaks, quotes and a possible byte-order mark
        $t = ([IO.File]::ReadAllText($f)).Trim().Trim([char]0xFEFF).Trim().Trim('"').Trim("'").Trim()
    } catch { Write-Host "Could not read $name ($($_.Exception.Message))" -ForegroundColor Yellow; continue }
    if (-not [string]::IsNullOrWhiteSpace($t)) { $token = $t; $tokenFile = $f; break }
}
if (-not $token) {
    Write-Host "No readable, non-empty access_token or access_token.txt in $dir." -ForegroundColor Red
    exit 1
}
Write-Host ("Using {0}; length {1} characters (the token itself is not shown)." -f (Split-Path $tokenFile -Leaf), $token.Length)

$user = $null
if (Test-Path $legacy) {
    try { $user = ([string]((Get-Content $legacy -Raw | ConvertFrom-Json).username)).Trim().ToLower() } catch { $user = $null }
}
if ([string]::IsNullOrWhiteSpace($user)) { $user = (Read-Host "Your Kaggle username").Trim().ToLower() }

if ($user -notmatch '^[a-z0-9][a-z0-9_-]{1,60}$') { Write-Host "That does not look like a Kaggle username. Nothing saved." -ForegroundColor Red; exit 1 }
if ([string]::IsNullOrWhiteSpace($token)) { Write-Host "The token file is empty. Nothing saved." -ForegroundColor Red; exit 1 }

modal secret create kaggle-credentials "KAGGLE_USERNAME=$user" "KAGGLE_API_TOKEN=$token" --force
if ($LASTEXITCODE -eq 0) { Write-Host "Saved the Modal secret 'kaggle-credentials' (username + access token)." -ForegroundColor Green } else { Write-Host "modal secret create failed. Nothing changed." -ForegroundColor Red }
Remove-Variable token, user -ErrorAction SilentlyContinue
