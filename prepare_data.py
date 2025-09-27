import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import chess


df=pd.read_csv(r'data\lichess_db_puzzle.csv.zst')
df=df[['FEN','Moves']]

def process_row(fen: str, moves: str):
    board = chess.Board(fen)
    f_local, m_local = [], []
    for idx, move_uci in enumerate(moves.split()):
        move = chess.Move.from_uci(move_uci)
        if idx % 2 == 1:
            f_local.append(board.fen())
            m_local.append(move_uci)
        board.push(move)
    return f_local, m_local

rows = df[["FEN", "Moves"]].values
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_row)(fen, moves) for fen, moves in tqdm(rows, total=len(rows))
)

f = [fen for f_local, _ in results for fen in f_local]
m = [move for _, m_local in results for move in m_local]

new_df=pd.DataFrame({'FEN':f,"Move":m})
new_df.to_csv('data/processed.csv',index=False)