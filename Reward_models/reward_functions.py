import torch

def calibrated_mean(mean,offset):
    return (mean+offset).clamp(0,1)

def pessimistic_reward(mean,uncertainty,offset,coefficient):
    if coefficient<0: raise ValueError('Pessimism coefficient must be nonnegative')
    return calibrated_mean(mean,offset)-coefficient*uncertainty
