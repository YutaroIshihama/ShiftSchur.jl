# ShiftSchur.jl
Restoring Shift Invariance in Dynamical Schur-Complement Sternheimer Response

## screen shot ps1

`$out="ShiftSchur_all_julia.txt"; Remove-Item $out -ErrorAction SilentlyContinue; Get-ChildItem -Path .\src,.\test -Recurse -Filter *.jl | Sort-Object FullName | ForEach-Object { "===== FILE: $($_.FullName.Substring((Get-Location).Path.Length+1)) =====`r`n"; Get-Content $_.FullName -Raw; "`r`n`r`n" } | Set-Content $out -Encoding utf8`
