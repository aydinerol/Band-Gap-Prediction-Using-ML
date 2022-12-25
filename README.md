# Band-Gap-Prediction-Using-ML

This repository contains the code and data for a machine learning project that predicts the band gap of materials using various features of the materials. The project was developed-benefiting codes in IntroMLLab[^1] and MAST-ML[^2][^3] in NanoHub-as part of a Master's thesis in Physics.

The repository contains the following files:

- main.py: This is the main script for the project. It contains the code for training and evaluating the machine learning model that predicts the band gap of a material from its chemical composition.
- BandGapPredict-Notebook.ipynb: A Jupyter notebook that contains the code and analysis for the project.
- requirements.txt: This file specifies the Python packages that are required for the project.
- data/* : A directory that contains the input data
- figures/* : A directory that contains the generated figures

## Requirements

In order to run this project, you will need the following:

- Python 3
- pandas==1.1.1
- numpy==1.19.1
- matplotlib==3.3.2
- scikit-learn==0.24.1

## Usage

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/aiostarex/Band-Gap-Prediction-Using-ML.git`
2. Navigate to the project directory: `cd Band-Gap-Prediction-Using-ML`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the project: `python main.py`


For more details about the project, please refer to the Jupyter notebook and the Master's thesis, which can be found in https://tez.yok.gov.tr/UlusalTezMerkezi/ under title "*Estimating the band gap of materials with machine learning methods*".

Please feel free to contact the author for any questions or comments about the project.

[^1]: BENJAMIN AFFLERBACH, Rundong Jiang, Josh Tappan, DANE MORGAN (2022), "Machine Learning Lab Module," https://nanohub.org/resources/intromllab. (DOI: 10.21981/CPNK-XE48)
[^2]: Ryan Jacobs, BENJAMIN AFFLERBACH (2022), "Materials Simulation Toolkit for Machine Learning (MAST-ML) tutorial," https://nanohub.org/resources/mastmltutorial. (DOI: 10.21981/WAYA-PF63)
[^3]: Jacobs, R., Mayeshiba, T., Afflerbach, B., Miles, L., Williams, M., Turner, M., Finkel, R., Morgan, D., "The Materials Simulation Toolkit for Machine Learning (MAST-ML): An automated open source toolkit to accelerate data- driven materials research", Computational Materials Science 175 (2020), 109544. https://doi.org/10.1016/j.commatsci.2020.109544
