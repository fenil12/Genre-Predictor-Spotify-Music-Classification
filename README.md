# 🎵 Spotify Genre Matrix Predictor 🎵

_By Team "To Be or Not to Be" (CS985MLDAGroup15)_

![Matrix Neo Gif](https://media.tenor.com/c-I5YMwtnLoAAAAS/matrix-neo.gif)

## 🌟 Team Spotlight:

- **Nishant Vimal** 
- **Fenil Patel** 
- **Rohit Satavekar** 
- **Chirdeep Singh Reen** 
- **Amit Pathak** 
- **Suditi Sharma** 

## 🎼 Project Essence:

Predict the rhythm of the genre using Spotify's rich musical dataset. Drawing data from [Kaggle](https://www.kaggle.com/competitions/cs9856-spotify-classification-problem-2023/data), the project is an ensemble of analytical processes aimed at predicting the pulse of the top music genres.

## 🛠 Dependencies:

Gear up by installing:

- pandas
- numpy
- matplotlib
- seaborn
- sklearn

To install, use the following pip command:
- pip install pandas numpy matplotlib seaborn scikit-learn

## 📊 Dataset Deep Dive:

With musical features like `tempo`, `speechiness`, and `acousticness` accentuating each track, the dataset paints a melodious picture. Every song is labelled with its respective genre, and the quest is to predict the genre.

### Features Explained:
- **ID**: Track's unique code.
- **title**: Song's name.
- **artist**: Maestro behind the melody.
- **top genre**: The very heart of the song (our target).
- Other sonic signatures like `tempo`, `speechiness`, `acousticness`, and more.

## 🚀 Approach:

1. **Data Tune-up**: Initial pre-processing involved removing any discordant missing values and visually inspecting the dataset's composition.
2. **Harmonizing Features**: Used one-hot encoding for artists, realizing that certain maestros have distinct genre imprints.
3. **Orchestrating Models**: Crafted a harmonious ensemble using a `VotingClassifier` that resonates with Logistic Regression, Random Forest, Support Vector Machines, and Extra Trees.
4. **Final Score**: Assessed the symphony using accuracy metrics and a confusion matrix.

## 🎤 Encore:

Our ensemble rendered a sonorous prediction, harmonizing perfectly with the genres. The minimal misclassifications echoed the model's finesse.

## 🎧 Listening Guide:

1. Sync the repository to your local ensemble.
2. Steer to the directory echoing with the notebook.
3. Initiate Jupyter Notebook and tune into the shared notebook.
4. Play the cells in harmony to experience the musical analysis.

