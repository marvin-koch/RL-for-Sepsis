"""
Forked from: https://github.com/twni2016/pomdp-baselines/blob/main/configs/rmdp/walker/rnn.yml
Modified to take inputs from our environment
"""


import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from rnn_utils import helpers as utl
from rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
import numpy as np
from .recurrent_critic import Critic_RNN
from .recurrent_actor import Actor_RNN


class ModelFreeOffPolicy_Separate_RNN(nn.Module):
    """
    Recurrent Actor and Recurrent Critic with separate RNNs
    Implemented for LSTM encoder
    """

    ARCH = "memory"
    Markov_Actor = False
    Markov_Critic = False

    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo_name,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        dqn_layers,
        policy_layers,
        device,
        rnn_num_layers=3,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        image_encoder_fn=lambda: None,
        **kwargs
    ):
        super().__init__()

        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.discount = 0.99

        self.algo = RL_ALGORITHMS[algo_name](**kwargs[algo_name], action_dim=action_dim)

        # Critic
        self.critic = Critic_RNN(
            obs_dim,
            action_dim,
            encoder,
            self.algo,
            action_embedding_size,
            observ_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            dqn_layers,
            rnn_num_layers,
            image_encoder=image_encoder_fn(),
        )
        self.critic_optimizer = Adam(self.critic.parameters(), lr=3e-5)

        # target network
        self.critic_target = deepcopy(self.critic)

        # Actor
        self.actor = Actor_RNN(
            obs_dim,
            action_dim,
            encoder,
            self.algo,
            action_embedding_size,
            observ_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            policy_layers,
            rnn_num_layers,
            image_encoder=image_encoder_fn(),  
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-3)

        # target networks
        self.actor_target = deepcopy(self.actor)

    @torch.no_grad()
    def get_initial_info(self):
        """
        Create inital state
        """
        return self.actor.get_initial_info()

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        device="cpu",
        deterministic=False,
        return_log_prob=False,
    ):
        """
        Get model's actor next action
        """
        prev_action = prev_action.unsqueeze(1)  # (1, B, dim)
        reward = reward.unsqueeze(1)  # (1, B, 1)
        obs = obs.unsqueeze(1)  # (1, B, dim)

        self.actor = self.actor.to(device)
        current_action_tuple, current_internal_state = self.actor.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            device=device
        )

        return current_action_tuple, current_internal_state

    @torch.no_grad()
    def act_target(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        device="cpu",
        deterministic=False,
        return_log_prob=False,
    ):
        """
        Get model's target actor next action
        """
        prev_action = prev_action.unsqueeze(1)  # (1, B, dim)
        reward = reward.unsqueeze(1)  # (1, B, 1)
        obs = obs.unsqueeze(1)  # (1, B, dim)

        self.actor_target = self.actor_target.to(device)
        current_action_tuple, current_internal_state = self.actor_target.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            device=device
        )

        return current_action_tuple, current_internal_state


    def forward(self, actions, next_actions, rewards, observs, dones, masks, scores, next_scores, loss):
        """
        Perform forward pass
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == observs.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == scores.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )
        num_valid = torch.clamp(masks.sum(), min=1.0) 

        #Critic loss
        (q1_pred, q2_pred), q_target = self.algo.critic_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            next_actions=next_actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            scores = scores,
            next_scores=next_scores,
            device = self.device
        )

        q1_pred = q1_pred.to(self.device)
        q2_pred = q2_pred.to(self.device)
        q_target = q_target.to(self.device)
        masks = masks.to(self.device)

        # masked Bellman error: masks (T,B,1) ignore the invalid entries (we only want to evalute one patient episode)
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks

        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        self.critic_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optimizer.step()

        #Actor loss
        policy_loss, log_probs, new_actions = self.algo.actor_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            score = scores,
            next_score=next_scores,
            actions=actions,
            rewards=rewards,
            device=self.device
        )

        # masked policy_loss
        policy_loss = (policy_loss * masks).sum() / num_valid
        act = actions[1:]*masks
        new_act = new_actions[:-1]*masks

        iv_param = 1
        vaso_param = 1
        policy_param = 0.2

        #MODIFIED LOSS FUNCTION
        if not loss == "none":
            lossparams = loss.split(",")
            iv_param = int(lossparams[0])
            vaso_param = int(lossparams[1])

        policy_loss = iv_param*(((act[:,:,0] - new_act[:,:,0]) ** 2).sum() / num_valid) + vaso_param*(((act[:,:,1] - new_act[:,:,1]) ** 2).sum() / num_valid) - policy_param * policy_loss
        
 
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        outputs = {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
        }

        #soft update
        self.soft_target_update()

        if log_probs is not None:
            with torch.no_grad():
                current_log_probs = (log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)

    def report_grad_norm(self):
        return {
            "q_grad_norm": utl.get_grad_norm(self.critic),
            "q_rnn_grad_norm": utl.get_grad_norm(self.critic.rnn),
            "pi_grad_norm": utl.get_grad_norm(self.actor),
            "pi_rnn_grad_norm": utl.get_grad_norm(self.actor.rnn),
        }


    def reform_reward(self, score, next_score, g1, g2):
        """
        Basic reward function
        """
        base_reward = g1* torch.tanh(score[:,2]-6)
        dynamic_reward = g2 * (next_score[:,2] - score[:,2])
        sofa_reward =  base_reward + dynamic_reward  
        sofa_reward = sofa_reward.unsqueeze(1)
        return sofa_reward

    def reform_reward_bias(self, score, next_score, g1, g2):
        """
        Reward function for biased score
        """
        base_reward = g1* torch.tanh(score-6)
        dynamic_reward = g2 * (next_score- score)
        sofa_reward =  base_reward + dynamic_reward  
        return sofa_reward



    def update(self, replay_buffer, loss, rew, rew_param):
        """
        Get episode for forward pass
        """
        state, action_c, next_action_c, next_state, reward, scores,next_scores, outcome,  not_done = replay_buffer.sample()

        batch_size = replay_buffer.batch_size

        if not self.algo.continuous_action:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        g1 = -0.2
        g2 = -0.125

        if not rew_param == "none":
            params_r = rew_param.split(",")           
            g1 = -float(params_r[0])
            g2 = -float(params_r[1])


        reward = self.reform_reward(scores, next_scores, g1, g2)

        #BIASED SOFA SCORE FOR REWARD
        if not rew == "none":
            state, action_c, next_action_c, next_state, reward, scores, sofa1, sofa2, sofa3, sofa4, sofa5, sofa6, next_scores, sofa1n, sofa2n, sofa3n, sofa4n, sofa5n, sofa6n,outcome, not_done = replay_buffer.sample(extra=True)
            r = rew.split(",")
            mod = r[0]
            bias = r[1]
            if mod == "simple" or mod == "both":
                if int(bias) < 1 or int(bias) > 6:
                  raise Exception("Invalid subscore")
                s = [0.8,0.8,0.8,0.8,0.8,0.8]
                s[int(bias) - 1] = 2
                sofa1_param,sofa2_param,sofa3_param,sofa4_param,sofa5_param,sofa6_param = s[0], s[1], s[2], s[3], s[4], s[5]
                bias_scores = (sofa1_param*sofa1 + sofa2_param*sofa2 + sofa3_param*sofa3 + sofa4_param*sofa4 + sofa5_param*sofa5 + sofa6_param*sofa6)

            s = [1,1,1,1,1,1]
            if mod == "both":
                if int(bias) < 1 or int(bias) > 6:
                  raise Exception("Invalid subscore")
                s = [0.8,0.8,0.8,0.8,0.8,0.8]
                s[int(bias) - 1] = 2
            sofa1_param,sofa2_param,sofa3_param,sofa4_param,sofa5_param,sofa6_param = s[0], s[1], s[2], s[3], s[4], s[5]
            bias_next_scores = (sofa1_param*sofa1n + sofa2_param*sofa2n + sofa3_param*sofa3n + sofa4_param*sofa4n + sofa5_param*sofa5n + sofa6_param*sofa6n)
            reward = self.reform_reward_bias(bias_scores, bias_next_scores, g1, g2)

  
        inverted_done = 1.0 - not_done

        # reshape data in to dimension (T+1, B, dim)
        actions, rewards, state, next_state, dones, scores, next_scores = action_c.unsqueeze(1), reward.unsqueeze(1), state.unsqueeze(1), next_state.unsqueeze(1), inverted_done.unsqueeze(1), ((scores[:,2]).reshape(-1,1).to(self.device)).unsqueeze(1), ((next_scores[:,2]).reshape(-1,1).to(self.device)).unsqueeze(1)
        next_actions = next_action_c.unsqueeze(1)
        
        observs = torch.cat((state[[0]], next_state), dim=0)  # (T+1, B, dim)

        actions = torch.cat(
            (ptu.zeros((1, 1, self.action_dim)).float().to(self.device), actions), dim=0
        )  # (T+1, B, dim)

        next_actions = torch.cat(
            (ptu.zeros((1, 1, self.action_dim)).float().to(self.device), next_actions), dim=0
        )  # (T+1, B, dim)

        rewards = torch.cat(
            (ptu.zeros((1, 1, 1)).float().to(self.device), rewards), dim=0
        )  # (T+1, B, dim)

        dones = torch.cat(
            (ptu.zeros((1, 1, 1)).float().to(self.device), dones), dim=0
        )  # (T+1, B, dim)

        scores = torch.cat(
            (ptu.zeros((1, 1, 1)).float().to(self.device), scores), dim=0
        )  # (T+1, B, dim)       

        next_scores = torch.cat(
            (ptu.zeros((1, 1, 1)).float().to(self.device), next_scores), dim=0
        )  # (T+1, B, dim)              

        not_done = not_done.cpu().numpy()
        episode_ends = torch.tensor(np.where(not_done == False)[0])
        episode_ends = episode_ends.to(self.device)
        masks = ptu.ones((batch_size, 1, 1))
        masks = masks.to(self.device)

        # We only want to keep one consecutive patient epsiode
        if not (episode_ends.numel()) == 0:
            e = episode_ends[0].cpu()
            idx = np.arange(e, batch_size)
            masks[idx, :, :] = 0.0

        return self.forward(actions, next_actions, rewards, observs, dones, masks, scores, next_scores, loss)