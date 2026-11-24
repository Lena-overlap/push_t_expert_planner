"""Run the unchanged baseline, then physical-rollout search on failed cases."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from expert_search import main
if __name__=='__main__': main(ROOT)
