import math
import numbers
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional
    pd = None

__all__ = [
    "generate_all_moves",
    "board_to_tensor",
    "get_dicts",
    "AlphaBetaMCTS",
    "run_alpha_beta_mcts",
    "PolicyIndexDataset",
]

COLORS = (chess.WHITE, chess.BLACK)
PIECES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def _bitboard_to_plane(bitboard: chess.Bitboard) -> np.ndarray:
    bits = np.array(list(f"{int(bitboard):064b}"), dtype=np.float32).reshape(8, 8)
    return np.fliplr(bits)


def generate_all_moves():
    l = list("abcdefgh")
    all_promotions = []
    for row in [7, 2]:
        for col in range(len(l)):
            for prom in ["n", "b", "r", "q"]:
                if 0 < col < 7:
                    for s in [-1, 0, 1]:
                        all_promotions.append(
                            f"{l[col]}{row}{l[col + s]}{row + 1 if row == 7 else row - 1}{prom}"
                        )
                if col == 0:
                    for s in [0, 1]:
                        all_promotions.append(
                            f"{l[col]}{row}{l[col + s]}{row + 1 if row == 7 else row - 1}{prom}"
                        )
                if col == 7:
                    for s in [-1, 0]:
                        all_promotions.append(
                            f"{l[col]}{row}{l[col + s]}{row + 1 if row == 7 else row - 1}{prom}"
                        )

    board = chess.Board(None)
    uci_moves = []
    for square in chess.SQUARES:
        board.clear()
        board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
        for move in board.legal_moves:
            uci_moves.append(move.uci())
        board.remove_piece_at(square)

    for square in chess.SQUARES:
        board.clear()
        board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
        for move in board.legal_moves:
            uci_moves.append(move.uci())
        board.remove_piece_at(square)
    return sorted(list(set(uci_moves + all_promotions)))


def board_to_tensor(board):
    planes = []
    for color in COLORS:
        for piece in PIECES:
            planes.append(_bitboard_to_plane(board.pieces(piece, color)))

    planes.append(np.full((8, 8), float(board.turn), dtype=np.float32))

    castling_plane = _bitboard_to_plane(board.castling_rights)
    planes.append(castling_plane)

    ep_plane = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        ep_plane[ep_rank, ep_file] = 1.0
    planes.append(np.flipud(ep_plane))

    stacked = np.stack(planes).astype(np.float32)
    return torch.from_numpy(stacked)


def get_dicts():
    all_moves = generate_all_moves()
    move_to_index = {key: value for value, key in enumerate(all_moves)}
    index_to_move = {value: key for key, value in move_to_index.items()}
    return move_to_index, index_to_move


class PolicyIndexDataset(Dataset):
    """Датасет для предобучения политики на готовом DataFrame или списке.

    Позволяет передавать целевые значения как готовые индексы или UCI-строки.
    Для строк необходимо предоставить словарь `move_to_index`.
    """

    def __init__(
        self,
        data: Union["pd.DataFrame", Sequence[Tuple[str, Union[str, int]]]],
        move_space_size: int,
        one_hot: bool = True,
        dtype: torch.dtype = torch.float32,
        fen_column: str = "fen",
        index_column: str = "move_index",
        move_to_index: Optional[Dict[str, int]] = None,
    ) -> None:
        if move_space_size <= 0:
            raise ValueError("move_space_size должен быть положительным числом")

        self.move_space_size = move_space_size
        self.one_hot = one_hot
        self.dtype = dtype
        self.move_to_index = move_to_index

        if pd is not None and isinstance(data, pd.DataFrame):
            if fen_column not in data.columns or index_column not in data.columns:
                raise KeyError(
                    f"DataFrame должен содержать колонки '{fen_column}' и '{index_column}'"
                )
            self.fens = data[fen_column].astype(str).tolist()
            self.targets = data[index_column].tolist()
        else:
            self.fens = []
            self.targets = []
            for item in data:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise ValueError(
                        "Каждый элемент последовательности должен быть парой (fen, move_idx)"
                    )
                fen, move_idx = item
                self.fens.append(str(fen))
                self.targets.append(move_idx)

        if len(self.fens) != len(self.targets):
            raise ValueError("Количество FEN и целевых значений не совпадает")

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fen = self.fens[item]
        target_raw = self.targets[item]
        move_idx = self._resolve_move_index(target_raw)

        board = chess.Board(fen)
        features = board_to_tensor(board)

        if self.one_hot:
            target = torch.zeros(self.move_space_size, dtype=self.dtype)
            target[move_idx] = 1.0
        else:
            target = torch.tensor(move_idx, dtype=torch.long)

        return features, target

    def _resolve_move_index(self, value: Union[str, int]) -> int:
        if isinstance(value, numbers.Integral):
            idx = int(value)
        elif isinstance(value, numbers.Real) and float(value).is_integer():
            idx = int(value)
        else:
            text_value = str(value)
            try:
                idx = int(text_value)
            except ValueError:
                if self.move_to_index is None:
                    raise ValueError(
                        "Не могу преобразовать целевое значение в индекс без move_to_index"
                    )
                idx = self.move_to_index.get(text_value)
                if idx is None:
                    raise KeyError(f"Ход '{text_value}' отсутствует в move_to_index")

        if idx < 0 or idx >= self.move_space_size:
            raise ValueError(
                f"Индекс хода {idx} выходит за границы пространства ({self.move_space_size})"
            )
        return idx


def _terminal_value(board: chess.Board) -> float:
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _make_cache_key(
    board: chess.Board, depth: int, color: int
) -> Tuple[str, bool, int, Optional[int], int, int]:
    return (
        board.board_fen(),
        board.turn,
        board.castling_rights,
        board.ep_square,
        depth,
        color,
    )


@dataclass
class MCTSNode:
    prior: float
    parent: Optional["MCTSNode"] = None
    children: Dict[chess.Move, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(
        self,
        board: chess.Board,
        policy: np.ndarray,
        move_to_index: Dict[str, int],
    ) -> None:
        if self.children:
            return

        moves = list(board.legal_moves)
        if not moves:
            return

        totals = []
        total_prob = 0.0
        for move in moves:
            idx = move_to_index.get(move.uci())
            prob = float(policy[idx]) if idx is not None else 0.0
            totals.append((move, prob))
            total_prob += prob

        if total_prob <= 0.0:
            uniform = 1.0 / len(moves)
            for move, _ in totals:
                self.children[move] = MCTSNode(prior=uniform, parent=self)
        else:
            for move, prob in totals:
                self.children[move] = MCTSNode(prior=prob / total_prob, parent=self)

    def select_child(self, c_puct: float) -> Tuple[chess.Move, "MCTSNode"]:
        best_score = -float("inf")
        best_move: Optional[chess.Move] = None
        best_child: Optional[MCTSNode] = None
        parent_visit_sqrt = math.sqrt(self.visit_count + 1)

        for move, child in self.children.items():
            exploration = (
                c_puct * child.prior * parent_visit_sqrt / (child.visit_count + 1)
            )
            score = child.value() + exploration
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        if best_child is None or best_move is None:
            raise RuntimeError("Failed to select child node in MCTS")
        return best_move, best_child

    def backup(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value


class AlphaBetaMCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        move_to_index: Dict[str, int],
        device: Optional[torch.device] = None,
        simulations: int = 128,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        alpha_beta_depth: int = 2,
    ) -> None:
        self.model = model
        self.move_to_index = move_to_index
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.alpha_beta_depth = alpha_beta_depth
        self._evaluation_cache: Dict[
            Tuple[str, bool, int, Optional[int], int, int], float
        ] = {}

    def search(self, board: chess.Board) -> MCTSNode:
        root = MCTSNode(prior=1.0)
        policy, _ = self._evaluate(board)
        root.expand(board, policy, self.move_to_index)
        self._apply_dirichlet_noise(root)

        for _ in range(self.simulations):
            scratch = board.copy(stack=False)
            path: List[MCTSNode] = [root]
            node = root

            while node.expanded() and node.children:
                move, node = node.select_child(self.c_puct)
                scratch.push(move)
                path.append(node)

            value = self._evaluate_leaf(scratch, node)

            for visited in reversed(path):
                visited.backup(value)
                value = -value

        return root

    def _apply_dirichlet_noise(self, node: MCTSNode) -> None:
        if self.dirichlet_alpha <= 0.0 or self.dirichlet_weight <= 0.0:
            return
        if not node.children:
            return

        keys = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(keys))
        for move, eta in zip(keys, noise):
            child = node.children[move]
            child.prior = (
                child.prior * (1.0 - self.dirichlet_weight)
                + eta * self.dirichlet_weight
            )

    def _evaluate_leaf(self, board: chess.Board, node: MCTSNode) -> float:
        if board.is_game_over():
            value = _terminal_value(board)
        else:
            policy, value = self._evaluate(board)
            node.expand(board, policy, self.move_to_index)
            if self.alpha_beta_depth > 0:
                value = self._alpha_beta(
                    board,
                    depth=self.alpha_beta_depth,
                    alpha=-float("inf"),
                    beta=float("inf"),
                    color=1,
                )
        return float(np.clip(value, -1.0, 1.0))

    def _evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        tensor = board_to_tensor(board).unsqueeze(0)
        if self.device is not None:
            tensor = tensor.to(self.device)
        with torch.no_grad():
            policy_logits, value_tensor = self.model(tensor)
        policy = F.softmax(policy_logits, dim=-1)[0].detach().cpu().numpy()
        value = float(value_tensor.squeeze().detach().cpu())
        return policy, value

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        color: int,
    ) -> float:
        key = _make_cache_key(board, depth, color)
        if key in self._evaluation_cache:
            return self._evaluation_cache[key]

        if board.is_game_over():
            terminal = _terminal_value(board)
            result = color * terminal
            self._evaluation_cache[key] = result
            return result

        policy, value = self._evaluate(board)
        if depth == 0:
            result = color * value
            self._evaluation_cache[key] = result
            return result

        moves = list(board.legal_moves)
        if not moves:
            result = color * value
            self._evaluation_cache[key] = result
            return result

        ordered = []
        for move in moves:
            idx = self.move_to_index.get(move.uci())
            prob = float(policy[idx]) if idx is not None else 0.0
            ordered.append((prob, move))
        ordered.sort(key=lambda item: item[0], reverse=True)

        best = -float("inf")
        for _, move in ordered:
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, -color)
            board.pop()

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        if best == -float("inf"):
            best = color * value

        self._evaluation_cache[key] = best
        return best


def _temperature_policy(node: MCTSNode, temperature: float) -> Dict[chess.Move, float]:
    if not node.children:
        return {}

    moves = list(node.children.keys())
    visits = np.array([node.children[m].visit_count for m in moves], dtype=np.float32)

    if temperature <= 1e-6:
        best_index = int(visits.argmax())
        return {moves[best_index]: 1.0}

    visits = np.power(visits, 1.0 / max(temperature, 1e-6))
    if visits.sum() == 0.0:
        prob = 1.0 / len(moves)
        return {move: prob for move in moves}

    visits /= visits.sum()
    return {move: float(weight) for move, weight in zip(moves, visits)}


def run_alpha_beta_mcts(
    board: chess.Board,
    model: torch.nn.Module,
    move_to_index: Dict[str, int],
    device: Optional[torch.device] = None,
    simulations: int = 256,
    alpha_beta_depth: int = 2,
    temperature: float = 1.0,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_weight: float = 0.25,
) -> Tuple[Optional[chess.Move], Dict[chess.Move, float], MCTSNode]:
    searcher = AlphaBetaMCTS(
        model=model,
        move_to_index=move_to_index,
        device=device,
        simulations=simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_weight=dirichlet_weight,
        alpha_beta_depth=alpha_beta_depth,
    )
    root = searcher.search(board)
    policy = _temperature_policy(root, temperature=temperature)
    if not policy:
        return None, {}, root

    best_move = max(policy.items(), key=lambda item: item[1])[0]
    return best_move, policy, root


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)
