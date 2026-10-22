from dataclasses import dataclass

@dataclass(frozen=True)
class Solution:
    name: str
    training: str
    reward: str
    decoding: str
    question: str

SOLUTIONS={
    'B0':Solution('B0','next_item_cross_entropy','none','top_k','Baseline sequential relevance'),
    'A1':Solution('A1','frozen_baseline','mean_reward','candidate_reranking','Can reranking replace RL?'),
    'A2':Solution('A2','reinforce_with_kl','mean_reward','top_k','Does adapting the policy help?'),
    'A3':Solution('A3','reinforce_with_kl','pessimistic_reward','top_k','Does uncertainty penalization help?'),
    'A4':Solution('A4','slate_reinforce_with_kl','pessimistic_reward_and_redundancy','next_k','Does joint slate optimization help?'),
}
