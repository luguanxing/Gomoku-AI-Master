from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room, leave_room
import ai
import torch
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__, template_folder="static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)

BOARD_SIZE = 15

# 全局房间与在线用户集合
GLOBAL_ROOM = "global"
clients = set()


class Game:
    def __init__(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = -1  # -1 和 1 轮流
        self.game_over = False
        self.winbegin = None
        self.winend = None
        self.last_move = None

    def reset(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = -1
        self.game_over = False
        self.winbegin = None
        self.winend = None
        self.last_move = None

    def make_move(self, x, y):
        # 非法或棋已结束
        if self.game_over or not self._in_bounds(x, y) or self.board[y][x] != 0:
            return False

        # 落子并判断胜利
        self.board[y][x] = self.current_player
        self.last_move = (x, y)

        if self._check_win(x, y):
            self.game_over = True
            return "win"

        # 切换轮次
        self.current_player = -1 * self.current_player
        return True

    def _in_bounds(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def _is_same(self, x, y, value):
        if not self._in_bounds(x, y):
            return False
        return self.board[y][x] == value

    def _check_win(self, x, y):
        # 四个方向：横、竖、主对角、次对角
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        color = self.board[y][x]

        for dx, dy in directions:
            count = 1
            # 记录连线两端，用于前端画高亮线
            self.winbegin = (x, y)
            self.winend = (x, y)

            # 正向
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if self._is_same(nx, ny, color):
                    count += 1
                    self.winbegin = (nx, ny)
                else:
                    break

            # 反向
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if self._is_same(nx, ny, color):
                    count += 1
                    self.winend = (nx, ny)
                else:
                    break

            if count >= 5:
                return True

        return False


# 全局唯一棋局
game = Game()


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    # 仅告知单个客户端连接成功
    socketio.emit("connection_status", "Connect Successfully", room=request.sid)


@socketio.on("join_game")
def handle_join_game():
    sid = request.sid
    join_room(GLOBAL_ROOM)
    clients.add(sid)

    # 把当前棋局发送给新加入的客户端
    socketio.emit(
        "game_start",
        {"board": game.board, "turn": game.current_player},
        room=sid,
    )

    # 向全体广播在线人数
    socketio.emit("connection_status", f"Online Users:{len(clients)}", room=GLOBAL_ROOM)


@socketio.on("restart_game")
def handle_restart_game():
    game.reset()
    socketio.emit(
        "game_start",
        {"board": game.board, "turn": game.current_player},
        room=GLOBAL_ROOM,
    )


@socketio.on("make_move")
def handle_make_move(data):
    if game.game_over:
        return

    x, y = data.get("x"), data.get("y")
    result = game.make_move(x, y)

    # 检查棋盘是否已满
    if all(cell != 0 for row in game.board for cell in row):
        result =  "stalemate"

    if result:
        # 广播最新棋盘
        socketio.emit(
            "update_board",
            {
                "board": game.board,
                "turn": game.current_player,
                "lastMove": {"row": y, "col": x},
            },
            room=GLOBAL_ROOM,
        )

        # 若胜利则广播结束
        if result == "win":
            socketio.emit(
                "game_over",
                {
                    "winner": game.current_player,  # 胜者为当前落子方
                    "line_begin": game.winbegin,
                    "line_end": game.winend,
                },
                room=GLOBAL_ROOM,
            )

        elif result == "stalemate":
            socketio.emit(
                "stalemate",
                {
                    "winner": 0,  # 平局
                    "line_begin": None,
                    "line_end": None,
                },
                room=GLOBAL_ROOM,
            )
    else:
        socketio.emit("message", "A piece is already placed there. Please select another position.", room=request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if sid in clients:
        clients.remove(sid)
    leave_room(GLOBAL_ROOM)

    # 广播最新在线人数
    socketio.emit("connection_status", f"Online Users:{len(clients)}", room=GLOBAL_ROOM)

# 使用ai模型跑
@socketio.on("ai_play")
def handle_ai_play(model_name):
    if game.game_over:
        return
    # 根据名称选择模型
    model_map = {
        "nn_model": nn_model,
        "dnn_model": dnn_model,
        "cnn_model": cnn_model,
        "rl_model" : rl_model,
        "voting" : None,
    }
    if model_name in model_map:
        model_key = model_name
    else:
        socketio.emit("message", "illegal model name!", room=GLOBAL_ROOM)
        return
    model = model_map[model_key]
    # 根据当前棋盘和玩家预测下一步
    board, current_player = game.board, game.current_player
    if model_name != "voting":
        row, col, reason = ai.predict_ai(model, board, current_player)
    else:
        # 调用所有模型，进行投票
        models = [nn_model, dnn_model, cnn_model, rl_model]
        votes = {}
        for m in models:
            row, col, _ = ai.predict_ai(m, board, current_player)
            key = (row, col)
            votes[key] = votes.get(key, 0) + 1
        # 选择票数最高的位置
        (row, col), _ = max(votes.items(), key=lambda x: x[1])
        reason = "voting"
    print("=" * 30)
    print("current board: \n", board)
    print("next player = ", current_player)
    print(f"chosen AI model: {model_name}")
    print(f"predict next position: row={row}, col={col}")
    print("=" * 30)
    socketio.emit(
        "ai_predict",
        {"x": col, "y": row, "rule": reason},
        room=GLOBAL_ROOM,
    )

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nn_model = ai.load_nn_model(model_path='models/nn_model.pth')
    dnn_model = ai.load_dnn_model(model_path='models/dnn_model.pth')
    cnn_model = ai.load_cnn_model(model_path='models/cnn_model.pth')
    rl_model = ai.load_cnn_model(model_path='models/cnn_model_rl_finetuned_frozen.pth')
    socketio.run(app, host="0.0.0.0", port=8080, debug=True, allow_unsafe_werkzeug=True)