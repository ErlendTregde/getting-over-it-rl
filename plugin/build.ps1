# Where the game is installed. There is deliberately no default: a guessed path
# is wrong for most people and fails as a confusing "cannot resolve
# Assembly-CSharp", which is worse than being told what to set.
if (-not $env:GOI_DIR) {
    Write-Host "GOI_DIR is not set." -ForegroundColor Red
    Write-Host 'Point it at your game folder, then reopen the shell:'
    Write-Host '  setx GOI_DIR "<drive>:\SteamLibrary\steamapps\common\Getting Over It"'
    exit 1
}
$plugins = Join-Path $env:GOI_DIR "BepInEx\plugins"

if (-not (Test-Path $plugins)) {
    Write-Host "No BepInEx plugins folder under GOI_DIR:" -ForegroundColor Red
    Write-Host "  $plugins" -ForegroundColor Red
    Write-Host "Install BepInEx 5.4.23.5 into the game folder first."
    exit 1
}

dotnet build
if ($LASTEXITCODE -ne 0) {
    Write-Host "build failed" -ForegroundColor Red
    exit 1
}

# The game holds the plugin DLL open while it runs, so the copy would fail and
# leave the old build in place. Say so instead of claiming success.
$game = Get-Process -Name "GettingOverIt" -ErrorAction SilentlyContinue
if ($null -ne $game) {
    Write-Host "Getting Over It is running (pid $($game.Id)) and holds GoiBridge.dll open." -ForegroundColor Yellow
    Write-Host "Close it, then re-run build.ps1." -ForegroundColor Yellow
    exit 1
}

try {
    Copy-Item "bin\Debug\netstandard2.0\GoiBridge.dll" $plugins -Force -ErrorAction Stop
    Write-Host "copied to plugins - start the game" -ForegroundColor Green
} catch {
    Write-Host "copy failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
