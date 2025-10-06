## Network Model example

This repository contains a minimal setup to reproduce the Zerlaut 2019 reults with JuliaSNN.

Use it as the starting point for the research projects.

### Installing

To reproduce this project, do the following:

Download this code base via git.  

Open a Julia console and do:

```
julia> using Pkg
julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
julia> Pkg.activate("path/to/this/project")
julia> Pkg.instantiate()
```

This will install all necessary packages for you to be able to run the scripts, and everything should work out of the box, including correctly finding local paths.

### Usage

The file `Zerlaut2019.jl` can be run via the interactive REPL in VSCode, in this case the simulation results are output on the Plot pane.

The code is structured such that it can also be run in a Jupyter notebook. To do this, you can install the jupyter notebook extension of VSCode and by right clicking on the file you should have the entry 
`open as Jupyter Notebook`