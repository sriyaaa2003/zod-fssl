# Installs your Kaggle credential for the 'kaggle' command line tool. Run it yourself; nothing is printed or logged.
#   cd C:\Users\sriya\zod-fssl
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_kaggle_token.ps1
#
# Two cases are handled:
#   1. You downloaded kaggle.json (older style) -> it is moved from Downloads to ~\.kaggle\kaggle.json
#   2. Kaggle showed you a single token string (newer style, shown once) -> you paste it at a hidden prompt and it
#      is saved as ~\.kaggle\access_token

$dir = Join-Path $HOME ".kaggle"
New-Item -ItemType Directory -Force $dir | Out-Null
$legacy = Join-Path $HOME "Downloads\kaggle.json"

if (Test-Path $legacy) {
    $target = Join-Path $dir "kaggle.json"
    Move-Item -Force $legacy $target
} else {
    $secure = Read-Host "No kaggle.json found. Paste your Kaggle API token (hidden)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    $value = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($value)) { Write-Host "Empty token. Nothing saved." -ForegroundColor Red; exit 1 }
    $target = Join-Path $dir "access_token"
    [IO.File]::WriteAllText($target, $value.Trim())
    Remove-Variable value, secure
}

# (an earlier version tried to restrict permissions here with a malformed icacls command and locked the file; removed.
#  Files in your profile folder are already private to your Windows account.)
Write-Host "Saved $target." -ForegroundColor Green
Write-Host "Test it with:   kaggle datasets list -s zod --max-size 1"
Write-Host "When the project is done, revoke the token on Kaggle: Settings > API."
