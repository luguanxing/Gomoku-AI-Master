# Summary
一个五子棋python网页应用，支持手动下棋和使用不同的神经网络模型进行下棋，模型由NN/DNN/CNN等模型根据棋谱进行训练生成 

A Python-based Gomoku web application that allows both manual play and AI-powered moves using different neural network models (e.g., NN, DNN, CNN), which are trained on historical game data
<img width="945" height="382" alt="image" src="https://github.com/user-attachments/assets/d666c376-f1e8-4f2e-a5fe-dc729b7702ba" />


<br>

# Background

- **Gomoku** (Five-in-a-Row, **五子棋**) is a strategy board game where two players take turns placing black and white pieces on a board, aiming to be the first to **_get five in a row_** horizontally, vertically, or diagonally.

- Gomoku strategy involves both **_offensive and defensive tactical patterns_**, such as creating open rows of three or four stones (known as "open three" or "open four") while simultaneously blocking the opponent’s potential winning lines.

- With the rise of artificial intelligence, machine learning models are increasingly being used to **predict the next _optimal move_**, analyze board states, and even challenge human players.


<br>

# System Architecture

## 🎯 Aims

- The main aim of this project is to **develop AI models capable of predicting the next best move in a Gomoku game**, given the current board state and player.
- This will be achieved using **neural networks** and potentially enhanced with advanced techniques.

## 🧱 Architecture

- 🔶 **Data Preparation**:  
  Jupyter Notebook (`.ipynb`) files responsible for generating and evaluating models.

- 🔷 **Web Interface**:  
  A simple HTML+CSS Gomoku web page for visualization of the current board.

- 🟢 **Web Server**:  
  A Python Flask web server responsible for handling web requests and calling models for predictions.

## 📁 Project Structure 
<img width="405" height="464" alt="image" src="https://github.com/user-attachments/assets/3cd41171-b537-48ec-ae27-0622b1199c8f" />


<br>

<br>


# Startup Instructions

1. Navigate to the `models_generator` directory and run all the Jupyter Notebook (`.ipynb`) files to generate the models.

2. Copy the generated model files into the `web_server/models` directory.

3. Rename the model files as needed.

4. Navigate to the `web_server` directory and run the Python web application.

> 💡 Make sure all dependencies are installed before running the notebooks and the web server.
