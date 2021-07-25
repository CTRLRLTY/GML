options = {
        "CXX": "clang++",
        "CCFLAGS": "-std=c++20 -g -O0",
        "CPPPATH": "headers/",
        "COMPILATIONDB_USE_ABSPATH": True
        }

env = Environment(**options)
env.Tool('compilation_db')
env.CompilationDatabase()

env.SharedLibrary('bin/gml', 'src/GML.cpp')
env.StaticLibrary('bin/gml', 'src/GML.cpp')
