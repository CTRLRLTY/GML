from glob import glob
import os

available_cpu = len(os.sched_getaffinity(0))
n_cpu = GetOption('num_jobs')
unit_test_cpp = glob('./test/*.cpp')

print("Available CPU: {}".format(available_cpu))

if GetOption('num_jobs') == 1:
    n_cpu = available_cpu
    SetOption('num_jobs', n_cpu)

print("Use --jobs={}".format(n_cpu))

options = {
        "CXX": "clang++",
        "CCFLAGS": "-std=c++20 -g -O0",
        "CPPPATH": "headers/",
        "COMPILATIONDB_USE_ABSPATH": True
        }

env = Environment(**options)
env.Tool('compilation_db')
env.CompilationDatabase()

env.Program('bin/implementation_test', [*unit_test_cpp])
