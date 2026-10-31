using DFTK
dftk_file = Base.find_package("DFTK")
println("DFTK.jl file: ", dftk_file)
pkg_root = joinpath(dirname(dftk_file), "..")
println("pkg_root: ", pkg_root)
response_dir = joinpath(pkg_root, "response")
println("response_dir: ", response_dir, " exists: ", isdir(response_dir))
# also try src/response
response_dir2 = joinpath(dirname(dftk_file), "response")
println("response_dir2: ", response_dir2, " exists: ", isdir(response_dir2))
for dir in [response_dir, response_dir2]
    if isdir(dir)
        for f in readdir(dir)
            println("  ", f)
        end
        chi0_path = joinpath(dir, "chi0.jl")
        if isfile(chi0_path)
            println("\n=== chi0.jl path: ", chi0_path, " ===")
            println(read(chi0_path, String))
        end
    end
end
