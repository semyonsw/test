<#
  Launcher for the Armenian call transcriber UI.

  Don't run this directly -- double-click "Transcribe Calls.bat" next to it.
  You can also drag a folder of recordings onto that .bat to browse that folder
  instead of this one.

  Override the interpreter with the STT_PYTHON environment variable if the venv
  ever moves.
#>
param([string]$CallsDir)

$ErrorActionPreference = 'Stop'

$Here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Server   = Join-Path $Here 'stt_server.py'
$Python   = if ($env:STT_PYTHON) { $env:STT_PYTHON } else { 'E:\nemo_stt_hy\venv\Scripts\python.exe' }
$HostAddr = '127.0.0.1'          # NB: $Host is reserved by PowerShell
$Ports    = 8000..8010

function Die([string]$msg) {
    Write-Host ''
    Write-Host "  $msg" -ForegroundColor Red
    Write-Host ''
    Read-Host '  Press Enter to close'
    exit 1
}

# Ports currently listening, via .NET so we don't depend on NetTCPIP cmdlets.
function Get-BusyPorts {
    [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().
        GetActiveTcpListeners() | ForEach-Object { $_.Port }
}

function ConvertTo-ComparablePath([string]$p) {
    if (-not $p) { return '' }
    return $p.TrimEnd('\').ToLowerInvariant()
}

# Returns the status object if OUR app answers on $p, else $null.
function Get-AppStatus([int]$p) {
    try {
        $r = Invoke-RestMethod "http://${HostAddr}:$p/api/status" -TimeoutSec 2 -UseBasicParsing
        if ($r.model -like '*stt_hy*') { return $r }
    } catch { }
    return $null
}

Write-Host ''
Write-Host '  Armenian Call Transcriber' -ForegroundColor Cyan
Write-Host '  ---------------------------------------------' -ForegroundColor DarkGray

if (-not (Test-Path $Python)) {
    Die ("Python environment not found:`n    $Python`n`n" +
         "  Set STT_PYTHON to your venv's python.exe if it lives somewhere else.")
}
if (-not (Test-Path $Server)) {
    Die "stt_server.py is not next to this launcher:`n    $Here"
}

if (-not $CallsDir) { $CallsDir = $Here }
if (-not (Test-Path $CallsDir -PathType Container)) { Die "Not a folder: $CallsDir" }
$CallsDir = (Resolve-Path $CallsDir).Path

# Reuse a running instance only if it is already serving THIS folder -- otherwise
# dragging a second folder onto the launcher would silently reopen the first one.
# Failing that, take the first genuinely free port.
$busy = Get-BusyPorts
$want = ConvertTo-ComparablePath $CallsDir
$port = $null
$existing = $null
foreach ($p in $Ports) {
    if ($busy -contains $p) {
        $s = Get-AppStatus $p
        if ($s -and (ConvertTo-ComparablePath $s.dir) -eq $want) {
            $port = $p; $existing = $s; break
        }
        continue                      # unrelated service, or our app on another folder
    }
    $port = $p; break
}
if (-not $port) { Die "No free port between $($Ports[0]) and $($Ports[-1])." }

$url = "http://${HostAddr}:$port/"

if ($existing) {
    Write-Host "  Already running" -ForegroundColor Yellow
    Write-Host "  recordings : $($existing.dir)"
    Write-Host "  address    : $url"
    Start-Process $url
    Start-Sleep -Seconds 1
    exit 0
}

Write-Host "  recordings : $CallsDir"
Write-Host "  address    : $url"
Write-Host ''

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:WANDB_MODE       = 'disabled'

$argLine = '"{0}" --host {1} --port {2} --dir "{3}"' -f $Server, $HostAddr, $port, $CallsDir
$proc = Start-Process -FilePath $Python -ArgumentList $argLine -NoNewWindow -PassThru

for ($i = 0; $i -lt 120; $i++) {
    if ($proc.HasExited) { Die "Server exited with code $($proc.ExitCode) -- see the messages above." }
    if (Get-AppStatus $port) { break }
    Start-Sleep -Milliseconds 500
}
if (-not (Get-AppStatus $port)) {
    try { $proc.Kill() } catch { }
    Die 'Server did not come up within 60 seconds.'
}

Write-Host '  Ready -- opening your browser.' -ForegroundColor Green
Write-Host '  Leave this window open. Close it (or press Ctrl+C) to stop the server.' -ForegroundColor DarkGray
Write-Host ''
Start-Process $url

$proc.WaitForExit()
