options = {
        "CXX": "clang++",
        "CCFLAGS": "-std=c++20 -g -O0",
        "COMPILATIONDB_USE_ABSPATH": True
        }

env = Environment(**options)
env.Tool('compilation_db')
env.CompilationDatabase()

env.SharedLibrary('bin/gml', 'src/main.cpp')
