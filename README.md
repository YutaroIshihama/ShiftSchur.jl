# ShiftSchur.jl
Restoring Shift Invariance in Dynamical Schur-Complement Sternheimer Response

## screen shot ps1

`$out="ShiftSchur_all_julia.txt"; Remove-Item $out -ErrorAction SilentlyContinue; Get-ChildItem -Path .\src,.\test -Recurse -Filter *.jl | Sort-Object FullName | ForEach-Object { "===== FILE: $($_.FullName.Substring((Get-Location).Path.Length+1)) =====`r`n"; Get-Content $_.FullName -Raw; "`r`n`r`n" } | Set-Content $out -Encoding utf8`

## Logs on Oct. 28

これがそのままSISCの Figure 1,2,3 になる。

- convergence_eta.png : k が増えると指数的に誤差が落ちる、ηが小さいと難しい
- rational_vs_poly.png : 同じkで rational が 1e5 倍良い
- greedy.png : greedy が等間隔より速く落ちる