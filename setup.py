from setuptools import setup

setup(name='semanticnetworks',
      version='1.0',
      description='Construction and analysis of semantic networks in text',
      url='https://github.com/nhchristianson/Math-text-semantic-networks',
      author='Nicolas Christianson',
      author_email='nicolas.christianson@gmail.com',
      license='CC-BY-4.0',
      packages=['semanticnetworks'],
      install_requires=[
        'nltk',
        'scipy',
        'numpy',
        'networkx',
        'pyenchant',
        'matplotlib',
        'traces',
        'ripser',
        'python-rake',
        'tqdm',
        'spacy',
        'pingouin',
        'seaborn',
        'pandas',
        'bctpy'
      ],
      zip_safe=False)
