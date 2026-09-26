<#
.SYNOPSIS
  Keep the HaluRISC FastAPI backend alive.

.DESCRIPTION
  The API process can disappear between sessions (job-object cleanup, port
  conflicts, transient crashes). This supervisor watches it and restarts it.

  Default mode runs a watchdog loop: health-check every few seconds, restart
  the API when it stops responding, and reclaim a stale listener on the port.

  Use -Once to start it a single time (no watchdog). Use -InstallTask to
  register a logon scheduled task so the API is always up on demo day.

.EXAMPLE
  pwsh -File scripts\serve_api.ps1
  pwsh -File scripts\serve_api.ps1 -Device cpu -Once
  pwsh -File scripts\serve_api.ps1 -InstallTask
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [ValidateSet("0", "1")][string]$Preload = "1",
    [int]$IntervalSeconds = 5,
    [string]$LogDir = (Join-Path $env:TEMP "halurisc"),
    [switch]$Once,
    [switch]$InstallTask,
    [switch]$UninstallTask
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$OutLog = Join-Path $LogDir "api-out.log"
$ErrLog = Join-Path $LogDir "api-err.log"
$SupervisorLog = Join-Path $LogDir "api-supervisor.log"
$TaskName = "HaluRISC API ($Port)"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $SupervisorLog -Value $line
    Write-Host $line
}

function Get-PortOwner {
    param([int]$LocalPort)
    $conn = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($conn) { return [int]$conn.OwningProcess }
    return 0
}

function Clear-PortOwner {
    param([int]$LocalPort, [string]$Reason)
    $owner = Get-PortOwner -LocalPort $LocalPort
    if ($owner -gt 0) {
        Write-Log "Stopping PID $owner on port $LocalPort ($Reason)"
        Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}

function Test-ApiHealth {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
        return ($r.status -eq "ok")
    } catch {
        return $false
    }
}

function Start-Api {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Python interpreter not found at $Python. Create the venv first."
    }
    $env:HALU_API_DEVICE = $Device
    $env:HALU_API_PRELOAD = $Preload

    Write-Log "Starting API on port $Port (device=$Device preload=$Preload)"
    $proc = Start-Process -FilePath $Python `
        -ArgumentList @("-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "$Port") `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -WindowStyle Hidden `
        -PassThru
    return $proc
}

function Start-ApiAndWait {
    Clear-PortOwner -LocalPort $Port -Reason "reclaim before start"

    $proc = Start-Api
    Write-Log "API process started (PID $($proc.Id)), waiting for health"

    $deadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $deadline) {
        if (Test-ApiHealth) {
            Write-Log "API healthy on http://127.0.0.1:$Port"
            return $true
        }
        if ($proc.HasExited) {
            Write-Log "API exited early with code $($proc.ExitCode). See $ErrLog"
            return $false
        }
        Start-Sleep -Seconds 3
    }
    Write-Log "API did not become healthy within 180 s"
    return $false
}

if ($UninstallTask) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Log "Removed scheduled task '$TaskName'"
    return
}

if ($InstallTask) {
    $script = Join-Path $PSScriptRoot "serve_api.ps1"
    $action = New-ScheduledTaskAction -Execute "pwsh" `
        -Argument "-NoProfile -WindowStyle Hidden -File `"$script`" -Device $Device -Preload $Preload"
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Log "Installed scheduled task '$TaskName' (runs at logon)"
    return
}

if ($Once) {
    if (Test-ApiHealth) {
        Write-Log "API already healthy on port $Port"
        return
    }
    [void](Start-ApiAndWait)
    return
}

Write-Log "Supervisor started (port $Port, interval ${IntervalSeconds}s, log $SupervisorLog)"

while ($true) {
    if (-not (Test-ApiHealth)) {
        Write-Log "Health check failed, restarting API"
        [void](Start-ApiAndWait)
    }
    Start-Sleep -Seconds $IntervalSeconds
}
