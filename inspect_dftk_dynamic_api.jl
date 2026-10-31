using DFTK
using LinearAlgebra
using Printf

println("=== DFTK dynamic-response API inspection ===")
println("Julia version = ", VERSION)
println("DFTK version  = ", pkgversion(DFTK))
println("DFTK path     = ", pathof(DFTK))
println()

function report_symbol(sym::Symbol)
    println("------------------------------------------------------------")
    println(sym)
    println("defined = ", isdefined(DFTK, sym))
    if isdefined(DFTK, sym)
        obj = getfield(DFTK, sym)
        try
            println("methods:")
            show(stdout, MIME("text/plain"), methods(obj))
            println()
        catch err
            println("methods unavailable: ", sprint(showerror, err))
        end
    end
end

for sym in (
    :apply_χ0,
    :solve_ΩplusK_split,
    :apply_potential_blochwave,
    :apply_kernel,
    :compute_δρ,
    :multiply_Ω,
    :multiply_ΩplusK,
    :solve_ΩplusK,
)
    report_symbol(sym)
end

root = dirname(pathof(DFTK))
println()
println("=== Source search root ===")
println(root)
println()

patterns = [
    "function apply_χ0",
    "apply_χ0(",
    "function solve_ΩplusK_split",
    "solve_ΩplusK_split(",
    "ΩplusK",
    "Sternheimer",
    "sternheimer",
    "frequency",
    "omega",
    "ω",
]

jl_files = String[]
for (dir, _, files) in walkdir(root)
    for f in files
        endswith(f, ".jl") && push!(jl_files, joinpath(dir, f))
    end
end

for pat in patterns
    println("=== SEARCH: ", repr(pat), " ===")
    nhit = 0
    for file in jl_files
        lines = try
            readlines(file)
        catch
            continue
        end
        for (iline, line) in pairs(lines)
            occursin(pat, line) || continue
            rel = relpath(file, root)
            @printf("%s:%d: %s\n", rel, iline, strip(line))
            nhit += 1
            nhit >= 40 && break
        end
        nhit >= 40 && break
    end
    nhit == 0 && println("(no hits)")
    println()
end

println("=== Inspection complete ===")
println("Next step: use this output to identify whether DFTK 0.8.1 already exposes")
println("a genuine frequency-dependent Sternheimer/chi0 operator. Do not guess an API.")
