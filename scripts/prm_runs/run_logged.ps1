# Run commands of the roadmap tools in order, appending all output to a log
# under <root>\prm\logs. Each argument after -Log is one command line (quoted):
#   "build ..." / "query ..." / "refine ..."   -> scripts\prm_pilot.py
#   "bridge validate ..." / "bridge birrt ..." -> scripts\prm_bridge.py
#   "carry stage1 ..." / "carry starts ..."    -> scripts\prm_carry_search.py
# Launch detached, from the repository root, e.g.
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File',
#       'scripts\prm_runs\run_logged.ps1','-Log','e2_refine_20261001.log',
#       '"refine --name B18_g2_n4000 --query all --workers 6"'
# ! Output is redirected through cmd: in Windows PowerShell 5.1 a native program
# ! writing to stderr under "*>>" becomes an error record and can stop the script.
param(
    [Parameter(Mandatory = $true)][string]$Log,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Commands
)
$env:HUSKY_BENCH_ROOT = "C:\Users\yijiangh\Insync\yijiang94817@gmail.com\Google Drive - Shared with me\2025-03 Husky Assembly\data_experiment\cdfm_planner_benchmark"
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$logPath = Join-Path $env:HUSKY_BENCH_ROOT "prm\logs\$Log"
foreach ($command in $Commands) {
    # Pick the script from the first word.
    if ($command.StartsWith("bridge ")) {
        $script = "scripts\prm_bridge.py"; $rest = $command.Substring(7)
    } elseif ($command.StartsWith("carry ")) {
        $script = "scripts\prm_carry_search.py"; $rest = $command.Substring(6)
    } else {
        $script = "scripts\prm_pilot.py"; $rest = $command
    }
    cmd /c "echo === $command >> `"$logPath`""
    cmd /c ".venv\Scripts\python.exe -u $script $rest >> `"$logPath`" 2>&1"
}
cmd /c "echo ALL DONE >> `"$logPath`""
