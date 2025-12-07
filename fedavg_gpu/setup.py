from setuptools import setup, Extension
import pybind11, numpy, os

cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

ext_modules = [
    Extension(
        "fedavg_gpu",
        sources=["binding.cpp"],           
        extra_objects=["fedavg_gpu.o"],     
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            os.path.join(cuda_home, "include"),
        ],
        library_dirs=[os.path.join(cuda_home, "lib64")],
        libraries=["cudart"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14"],
    )
]

setup(
    name="fedavg_gpu",
    version="0.1",
    ext_modules=ext_modules,
)
