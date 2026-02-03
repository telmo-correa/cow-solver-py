"""Build script for Cython extensions.

Run with: python setup_cython.py build_ext --inplace
"""

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    extensions = [
        Extension(
            "solver.math.fixed_point_cy",
            ["solver/math/fixed_point_cy.pyx"],
        ),
        Extension(
            "solver.safe_int_cy",
            ["solver/safe_int_cy.pyx"],
        ),
    ]

    setup(
        name="cow-solver-cython",
        packages=[],  # Prevent auto-discovery
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
            },
        ),
    )
except ImportError:
    print("Cython not installed. Run: pip install cython")
    raise
