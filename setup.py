from setuptools import setup

setup(
    name='rl4mixed',
    version='0.1',
    python_requires='>=3.9.0',
    py_modules=[],
    install_requires=[
        "gurobipy==10.0.0", 
        "pandas>=1.5.0", 
        "matplotlib",
        "torch>=1.13.0",
        "neptune",
        "hydra-core",
        "hydra-joblib-launcher",
        "tqdm",
        "pillow"
    ],
    entry_points={
        'console_scripts': [
            'train-actor = rl4mixed.trainer:main',
            'solve-exact = rl4mixed.gurobi.main:main',
            'sim-testdata = rl4mixed.validation:main',
        ],
    },
)