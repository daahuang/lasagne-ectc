from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("ctc_fast_noblank_mask_semi_stay", ["ctc_fast_noblank_mask_semi_stay.pyx"])]
)
