<#
.SYNOPSIS
  Manage and keep alive the HaluRISC FastAPI backend.

.DESCRIPTION
  The API process can disappear between sessions (job-object cleanup, port
  conflicts, transient crashes). This script watches it and restarts it.

  It always uses the project virtual environment interpreter, so a second copy
  started from a different Python cannot steal the port. A named mutex prevents
  two supervisors from fighting over the same port.

  Modes:
    (default)      run the watchdog loop
    -Once          start it a single time, no watchdog
    -Stop          stop every HaluRISC API process and release the port
    -InstallTask   register a logon scheduled task
    -UninstallTask remove that scheduled task

.EXAMPLE
  pwsh -File scripts\serve_api.ps1
  pwsh -File scripts\serve_api.ps1 -Once -Device cpu
  pwsh -File scripts\serve_api.ps1 -Stop
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
    [switch]$Stop,
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
$AppPattern = "*uvicorn*src.api.main*"

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

function Get-ApiProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like $AppPattern }
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
    Write-Log "Started $Python (PID $($proc.Id), device=$Device, preload=$Preload)"

    $deadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $deadline) {
        if (Test-ApiHealth) {
            $owner = Get-PortOwner -LocalPort $Port
            Write-Log "API healthy on http://127.0.0.1:$Port (serving PID $owner)"
            return $true
        }
        if ($proc.HasExited) {
            $tail = (Get-Content -LiteralPath $ErrLog -Tail 20 -ErrorAction SilentlyContinue) -join "`n"
            if ($tail -match "10048") {
                Write-Log "Port $Port was already in use (WinError 10048). Backing off."
                Start-Sleep -Seconds 5
                if (Test-ApiHealth) {
                    Write-Log "Another instance is already healthy, leaving it alone."
                    return $true
                }
                Clear-PortOwner -LocalPort $Port -Reason "reclaim after 10048"
                $proc = Start-Api
                Write-Log "Retrying with PID $($proc.Id)"
                continue
            }
            Write-Log "API exited early with code $($proc.ExitCode). See $ErrLog"
            return $false
        }
        Start-Sleep -Seconds 3
    }
    Write-Log "API did not become healthy within 180 s"
    return $false
}

function Stop-AllApi {
    $stopped = 0

    # Stop any watchdog first, so it cannot restart the API we are about to kill.
    foreach ($p in Get-CimInstance Win32_Process -Filter "Name='pwsh.exe'" -ErrorAction SilentlyContinue) {
        if ($p.ProcessId -eq $PID) { continue }
        if ($p.CommandLine -like "*-File*serve_api.ps1*" -and $p.CommandLine -notlike "*-Stop*") {
            Write-Log "Stopping supervisor PID $($p.ProcessId)"
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }

    $owner = Get-PortOwner -LocalPort $Port
    if ($owner -gt 0) {
        Write-Log "Stopping listener PID $owner on port $Port"
        Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
        $stopped++
    }
    foreach ($p in Get-ApiProcesses) {
        Write-Log "Stopping API process PID $($p.ProcessId)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        $stopped++
    }
    Start-Sleep -Seconds 2
    if (Get-PortOwner -LocalPort $Port) {
        Write-Log "Warning: port $Port is still held"
    } else {
        Write-Log "Port $Port is free (stopped $stopped process(es))"
    }
}

if ($Stop) {
    Stop-AllApi
    return
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

# Watchdog mode: one supervisor per port.
$createdNew = $false
$mutex = [System.Threading.Mutex]::new($true, "Local\HaluRISC.Api.Supervisor.$Port", [ref]$createdNew)
if (-not $createdNew) {
    Write-Log "A supervisor for port $Port is already running. Exiting."
    $mutex.Dispose()
    return
}

try {
    Write-Log "Supervisor started (port $Port, interval ${IntervalSeconds}s, interpreter $Python)"
    while ($true) {
        if (-not (Test-ApiHealth)) {
            Write-Log "Health check failed, restarting API"
            [void](Start-ApiAndWait)
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
