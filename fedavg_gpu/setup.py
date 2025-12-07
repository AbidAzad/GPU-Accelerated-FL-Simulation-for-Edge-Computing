from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fedavg_gpu",
        sources=["binding.cpp", "fedavg_gpu.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"]
        },
    ),
]

setup(
    name="fedavg_gpu",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
