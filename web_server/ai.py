import torch
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 15

class SimpleNet(nn.Module):
    def __init__(self, input_dim=226, output_dim=225):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
    def forward(self, x):
        # x: (batch, 226), board_flat: (batch, 225)
        board_flat = x[:, 1:]  # (batch, 225)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)  # (batch, 225)
        # mask非法落子
        mask = (board_flat == 0)
        logits = logits.masked_fill(~mask, -1e9)
        return logits

class ResidualDNNNet(nn.Module):
    def __init__(self, input_dim=226, output_dim=225):
        super(ResidualDNNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, output_dim)
        # 用于残差连接，将输入x升维到128
        self.res_fc = nn.Linear(input_dim, 128)
    def forward(self, x):
        board_flat = x[:, 1:]  # (batch, 225)
        x1 = F.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)
        x2 = F.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        # 残差连接：输入x升维后与x2相加
        res = self.res_fc(x)
        x2 = x2 + res
        logits = self.fc3(x2)  # (batch, 225)
        # mask非法落子
        mask = (board_flat == 0)
        logits = logits.masked_fill(~mask, -1e9)
        return logits

class ResidualCNNNet(nn.Module):
    def __init__(self, input_channels=2, board_size=15, output_dim=225):
        super(ResidualCNNNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.1)
        self.res_conv = nn.Conv2d(input_channels, 64, kernel_size=1)
        self.fc = nn.Linear(board_size * board_size * 64, output_dim)
    def forward(self, x):
        # x: (batch, 226) -> (batch, 2, 15, 15)
        player = x[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 15, 15)
        board = x[:, 1:].reshape(-1, 1, 15, 15)
        x_cnn = torch.cat([player, board], dim=1)
        out = F.relu(self.bn1(self.conv1(x_cnn)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.dropout(out)
        # 残差连接
        res = self.res_conv(x_cnn)
        out = out + res
        out = out.view(out.size(0), -1)
        logits = self.fc(out)  # (batch, 225)
        # mask非法落子
        board_flat = x[:, 1:]
        mask = (board_flat == 0)
        logits = logits.masked_fill(~mask, -1e9)
        return logits

def load_nn_model(model_path, input_dim=226, output_dim=225):
    model = SimpleNet(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_dnn_model(model_path, input_dim=226, output_dim=225):
    model = ResidualDNNNet(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_cnn_model(model_path, input_channels=2, board_size=15, output_dim=225):
    model = ResidualCNNNet(input_channels=input_channels, board_size=board_size, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_ai(model, board_state, player):
    board = np.array(board_state)
    empties = [(int(r), int(c)) for r, c in np.argwhere(board == 0)]
    opp = -1 if player == 1 else 1

    # 预先算一次 logits（用于所有规则的tie-break）
    pred_logits = _predict_logits_for_board(model, board, player)

    # 规则兜底1：我方立即获胜
    cands = [(r, c) for (r, c) in empties if _has_immediate_win(board, r, c, player)]
    if cands:
        r, c = _pick_best_by_logits(pred_logits, cands)
        return r, c, ("win_5(logits)" if len(cands) > 1 else "win_5")

    # 规则兜底2：堵对方立即获胜
    cands = [(r, c) for (r, c) in empties if _has_immediate_win(board, r, c, opp)]
    if cands:
        r, c = _pick_best_by_logits(pred_logits, cands)
        return r, c, ("block_win_5(logits)" if len(cands) > 1 else "block_win_5")

    # 规则兜底3：我方活四
    cands = [(r, c) for (r, c) in empties if _is_open_four_move(board, r, c, player)]
    if cands:
        r, c = _pick_best_by_logits(pred_logits, cands)
        return r, c, ("live_4(logits)" if len(cands) > 1 else "live_4")

    # 规则兜底4：堵对方活四
    cands = [(r, c) for (r, c) in empties if _is_open_four_move(board, r, c, opp)]
    if cands:
        r, c = _pick_best_by_logits(pred_logits, cands)
        return r, c, ("block_live_4(logits)" if len(cands) > 1 else "block_live_4")

    # 规则兜底5：我方双活三
    cands = [(r, c) for (r, c) in empties if _is_double_three_move(board, r, c, player)]
    if cands:
        r, c = _pick_best_by_logits(pred_logits, cands)
        return r, c, ("double_3(logits)" if len(cands) > 1 else "double_3")

    # 规则兜底6：堵对方双活三
    cands = [(r, c) for (r, c) in empties if _is_double_three_move(board, r, c, opp)]
    if cands:
        r, c = _pick_best_by_logits(pred_logits, cands)
        return r, c, ("block_double_3(logits)" if len(cands) > 1 else "block_double_3")

    # 规则兜底4.5：我方 4+3（冲四 + 活三）
    cands = [(r, c) for (r, c) in empties if _is_four_plus_three_move(board, r, c, player)]
    if cands:
        r, c = _pick_best_by_logits(pred_logits, cands)
        return r, c, ("four_plus_three(logits)" if len(cands) > 1 else "four_plus_three")

    # 规则兜底4.6：堵对方 4+3
    cands = [(r, c) for (r, c) in empties if _is_four_plus_three_move(board, r, c, opp)]
    if cands:
        r, c = _pick_best_by_logits(pred_logits, cands)
        return r, c, ("block_four_plus_three(logits)" if len(cands) > 1 else "block_four_plus_three")

    # 若无命中上述规则，直接走模型argmax
    best_idx = int(np.argmax(pred_logits))
    pred_row, pred_col = divmod(best_idx, BOARD_SIZE)
    return pred_row, pred_col, "ai_prediction"


def _predict_logits_for_board(model, board, player) -> np.ndarray:
    """
    返回当前局面下，模型对所有 225 个点（非法点已被模型内部mask为-1e9）的logits。
    shape: (225,)
    """
    sample_X = np.concatenate([[player], board.flatten()])
    with torch.no_grad():
        t = torch.tensor(sample_X, dtype=torch.float32).unsqueeze(0)  # (1, 226)
        logits = model(t).squeeze(0)  # (225,)
    return logits.detach().cpu().numpy()


def _pick_best_by_logits(pred_logits: np.ndarray, candidates):
    """
    candidates: [(r, c), ...]
    在给定候选中，用 logits 选分数最高的那个点。
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    idxs = [r * BOARD_SIZE + c for (r, c) in candidates]
    scores = pred_logits[idxs]
    best_local = int(np.argmax(scores))
    return candidates[best_local]


def _to_ai_cell(v: int) -> int:
    # 1 -> -1, 2 -> 1, 0/others -> 0
    return -1 if v == 1 else 1 if v == 2 else 0

def _has_immediate_win(b, r, c, pid):
    if b[r, c] != 0:
        return False
    # Check 4 directions: |, ---, \, /
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dr, dc in dirs:
        cnt = 1  # include the placed stone at (r, c)
        i, j = r + dr, c + dc
        while 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and b[i, j] == pid:
            cnt += 1
            i += dr
            j += dc
        i, j = r - dr, c - dc
        while 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and b[i, j] == pid:
            cnt += 1
            i -= dr
            j -= dc
        if cnt >= 5:
            return True
    return False

def _dir_line_str(b, r, c, dr, dc, pid):
    # Build a 9-length string centered at (r,c) along direction (dr,dc)
    # Encoding: '1' = pid (including the hypothetical move at center), '0' = empty, '2' = opponent or border
    chars = []
    for k in range(-4, 5):
        i, j = r + k * dr, c + k * dc
        if i < 0 or i >= BOARD_SIZE or j < 0 or j >= BOARD_SIZE:
            chars.append('2')
        elif k == 0:
            chars.append('1')  # hypothetical stone
        else:
            v = b[i, j]
            if v == 0:
                chars.append('0')
            elif v == pid:
                chars.append('1')
            else:
                chars.append('2')
    return ''.join(chars)

def _count_occurrences(s: str, pat: str) -> int:
    cnt, start = 0, 0
    while True:
        i = s.find(pat, start)
        if i == -1:
            break
        cnt += 1
        start = i + 1  # allow overlapping
    return cnt

def _is_open_four_move(b, r, c, pid) -> bool:
    if b[r, c] != 0:
        return False
    # .XXXX. in any direction
    for dr, dc in [(1,0), (0,1), (1,1), (1,-1)]:
        s = _dir_line_str(b, r, c, dr, dc, pid)
        if '011110' in s:
            return True
    return False

def _has_open_three_dir(b, r, c, dr, dc, pid) -> bool:
    """
    判断在方向 (dr, dc) 上，当前在 (r, c) 落子 pid 后，
    是否存在“再下一步即可形成活四(011110)”的机会。
    这等价于该方向上存在一个“活三”。
    """
    if b[r, c] != 0:
        return False

    s = _dir_line_str(b, r, c, dr, dc, pid)

    # 若当前就已经是活四，这个方向不算作活三（更高优先级的规则会先处理活四）
    if '011110' in s:
        return False

    # 枚举把该方向线上任一空位再填成己子，是否能出现活四
    for idx, ch in enumerate(s):
        if ch != '0':
            continue
        s2 = s[:idx] + '1' + s[idx+1:]
        if '011110' in s2:
            return True
    return False


def _is_double_three_move(b, r, c, pid) -> bool:
    """
    是否在 (r, c) 落子 pid 后，至少有两个不同方向出现“活三”（即再下一步可成活四）。
    """
    if b[r, c] != 0:
        return False
    cnt = 0
    for dr, dc in [(1,0), (0,1), (1,1), (1,-1)]:
        if _has_open_three_dir(b, r, c, dr, dc, pid):
            cnt += 1
            if cnt >= 2:
                return True
    return False

def _count_win_points_in_dir(b, r, c, dr, dc, pid) -> int:
    """
    After hypothetically placing pid at (r,c), count how many single empty cells
    on this direction can be filled to make '11111'.
    """
    s = _dir_line_str(b, r, c, dr, dc, pid)
    # Immediate win would already be handled by higher-priority rule
    if '11111' in s:
        return 2
    cnt = 0
    for idx, ch in enumerate(s):
        if ch != '0':
            continue
        s2 = s[:idx] + '1' + s[idx+1:]
        if '11111' in s2:
            cnt += 1
    return cnt

def _has_blocked_four_dir(b, r, c, dr, dc, pid) -> bool:
    """
    '冲四' on this direction: exactly one winning point (not open four).
    """
    s = _dir_line_str(b, r, c, dr, dc, pid)
    if '011110' in s:  # open four excluded
        return False
    return _count_win_points_in_dir(b, r, c, dr, dc, pid) == 1

def _is_four_plus_three_move(b, r, c, pid) -> bool:
    """
    4+3: at least one direction is blocked-four AND another direction is open-three.
    """
    if b[r, c] != 0:
        return False
    dirs = [(1,0), (0,1), (1,1), (1,-1)]
    blocked_dirs = {i for i, (dr, dc) in enumerate(dirs) if _has_blocked_four_dir(b, r, c, dr, dc, pid)}
    if not blocked_dirs:
        return False
    open3_dirs = {i for i, (dr, dc) in enumerate(dirs) if _has_open_three_dir(b, r, c, dr, dc, pid)}
    # Require different directions
    return len(open3_dirs - blocked_dirs) > 0