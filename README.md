# JulianSchwingerModel
Lattice Schwinger model with two flavours.
Based on https://github.com/urbach/schwinger and rewritten in Julia.
The lecture slides for the original tutorial can be found at 
http://theor.jinr.ru/~diastp/summer11/lectures/Urbach-2.pdf

# Using this code

This code has been tested on Mac OSX, but should work for any architechure
and operating system that uses Julia. You will need julia v1.0.1 to run this code,
it can be downloaded here:

   https://julialang.org/downloads/

Once you have julia downloaded and compiled, open the application and type
a closing square bracket "]" and enter the command:

  dev https://github.com/ylin910095/JulianSchwingerModel.jl.git

This will download the JulianSchwingerModel git repository into the directory

     ~/.julia/dev

You may now import the entire contents of this repository into your julia
scripts using the julia command:

	"using JulianSchwingerModel" or "import JulianSchwingerModel"

An example has been provided in

   example/pion_corr.jl

To run this package from the Terminal, you must first make the julia executable 
visible to your environment. To do this, create a soft link in your local bin
directory, for example:

   ln -s /Applications/Julia-1.0.1.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia

Then change directory to 'example' and run the command:

     cd example
     julia pion_corr.jl


