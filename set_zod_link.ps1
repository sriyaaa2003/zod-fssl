# Stores the ZOD download link from the Zenseact approval email as the Modal secret 'zod-download'.
# Run it yourself in a terminal; the link is typed at the prompt and is never written to a file.
#   cd C:\Users\sriya\zod-fssl
#   powershell -ExecutionPolicy Bypass -File .\set_zod_link.ps1

$url = Read-Host "Paste the ZOD download link from the Zenseact email (right-click 'this link' > Copy link address)"
$url = $url.Trim().Trim('"').Trim("'")

if ($url -notmatch '^https://([a-z0-9-]+\.)?dropbox\.com/') {
    Write-Host "That does not look like a Dropbox link (expected https://www.dropbox.com/...). Nothing was saved." -ForegroundColor Red
    exit 1
}

modal secret create zod-download "ZOD_URL=$url" --force
if ($LASTEXITCODE -eq 0) {
    Write-Host "Saved. The secret 'zod-download' now holds your link." -ForegroundColor Green
} else {
    Write-Host "modal secret create failed. Nothing changed." -ForegroundColor Red
}
Remove-Variable url
