from setuptools import find_packages, setup

setup(
    name='hgraph2graph',
    version='0.1',
    author='Wengong Jin',
    packages=find_packages(include=['hgraph', 'hgraph.*']),  # Укажите основной пакет
    install_requires=[
        'torch',
        'rdkit',
        'tqdm',
        # Добавьте другие зависимости, если они есть
    ],
)