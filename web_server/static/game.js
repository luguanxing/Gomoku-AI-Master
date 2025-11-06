const socket = io();
const boardSize = 15;
let board = [];
let currentPlayer = -1; // -1: 黑棋, 1: 白棋
let gameOver = false;
let lastMove = null;

const connectionStatusElement = document.getElementById('connection-status');
const boardElement = document.getElementById('board');
const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const hintElement = document.getElementById('hint');
const restartButton = document.getElementById('restart');
const aiButton = document.getElementById('ai_run');
const aiModel = document.getElementById('ai_model');

// 初始化棋盘
function initBoard() {
    board = Array.from({ length: boardSize }, () =>
        Array.from({ length: boardSize }, () => 0)
    );

    boardElement.innerHTML = '';
    for (let i = 0; i < boardSize; i++) {
        for (let j = 0; j < boardSize; j++) {
            const cell = document.createElement('div');
            cell.dataset.row = i;
            cell.dataset.col = j;
            cell.addEventListener('click', handleCellClick);
            boardElement.appendChild(cell);
        }
    }
}

// 处理点击事件
function handleCellClick(event) {
    if (gameOver) return;
    const row = parseInt(event.target.dataset.row);
    const col = parseInt(event.target.dataset.col);

    if (board[row][col] !== 0) {
        showMessage("A piece is already placed there. Please select another position.");
        return;
    }

    socket.emit('make_move', { x: col, y: row });
}

// 显示提示信息
let messageTimer = null; // 用于保存定时器句柄

function showMessage(message) {
    messageElement.textContent = message;
    messageElement.classList.add('show');
    if (messageTimer) {
        clearTimeout(messageTimer); // 清除前一个定时器
    }
    messageTimer = setTimeout(() => {
        messageElement.classList.remove('show');
        messageTimer = null; // 清空句柄
    }, 3000); // 3秒后隐藏提示信息
}

// 显示提示信息2
let hintTimer = null; // 用于保存定时器句柄

function showHint(message) {
    hintElement.textContent = message;
    hintElement.classList.add('show_hint');
    if (hintTimer) {
        clearTimeout(hintTimer); // 清除前一个定时器
    }
    hintTimer = setTimeout(() => {
        hintElement.classList.remove('show_hint');
        hintTimer = null; // 清空句柄
    }, 3000); // 3秒后隐藏提示信息
}

// 更新棋盘状态
function updateBoard(newBoard, turn, lastMovePosition) {
    board = newBoard;
    currentPlayer = turn;
    
    boardElement.childNodes.forEach((cell, index) => {
        const row = Math.floor(index / boardSize);
        const col = index % boardSize;
        const value = board[row][col];
        
        cell.className = '';
        if (value === -1) {
            cell.classList.add('black');
        } else if (value === 1) {
            cell.classList.add('white');
        }

        if (lastMove && lastMove.row === row && lastMove.col === col) {
            cell.classList.remove('last-move');
        }
        if (lastMovePosition && lastMovePosition.row === row && lastMovePosition.col === col) {
            cell.classList.add('last-move');
        }
    });

    lastMove = lastMovePosition;
    
    statusElement.textContent = `Current Player: ${currentPlayer === -1 ? 'Black' : 'White'}`;
}

// 游戏结束处理
function handleGameOver(winner, lineBegin, lineEnd) {
    gameOver = true;
    statusElement.textContent = `Gameover，${winner === -1 ? 'Black' : 'White'} wins！`;
}

// 重新开始游戏
function restartGame() {
    socket.emit("restart_game");
}

// WebSocket事件监听
socket.on('connect', () => {
    socket.emit('join_game');
});

socket.on('game_start', (data) => {
    initBoard();
    updateBoard(data.board, data.turn);
    gameOver = false;
});

socket.on('update_board', (data) => {
    updateBoard(data.board, data.turn, data.lastMove);
});

socket.on('game_over', (data) => {
    handleGameOver(data.winner, data.line_begin, data.line_end);
});

socket.on('stalemate', () => {
    statusElement.textContent = 'The game is a draw!';
    gameOver = true;
});

socket.on("connection_status", (data) => {
	connectionStatusElement.textContent = data;
});

socket.on("message", (data) => {
    showMessage(data);
});

// 接收后端AI的预测信息
socket.on("ai_predict", ({ x, y, rule }) => {
    console.log('-' * 20)
    console.log("AI move to:", x, y);
    console.log(rule)
    showHint("Using rule: " + rule);
    const cell = boardElement.querySelector(`[data-row="${y}"][data-col="${x}"]`);
    if (cell) {
        cell.click()
    } else {
        console.warn("AI target cell not found:", { x, y });
    }
});

// 后端ai下棋
function aiPlay() {
    console.log("AI play with model:", aiModel.value);
    socket.emit("ai_play", aiModel.value);
}

// 绑定事件监听器
restartButton.addEventListener('click', restartGame);
aiButton.addEventListener('click', aiPlay)

// 初始化游戏
initBoard();
