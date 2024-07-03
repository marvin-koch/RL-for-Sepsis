"""
Forked from: https://github.com/twni2016/Memory-RL
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

from .gpt_default import get_config

from .gpt2_vanilla import GPT2


class ModelFreeOffPolicy_Shared_RNN(nn.Module):
    """
    Recurrent Actor and Recurrent Critic with shared RNN
    Implemented for Transformer encoder using GPT
    We find `freeze_critic = True` can prevent degradation shown in https://github.com/twni2016/pomdp-baselines
    """

    ARCH = "memory"

    def __init__(
        self,
        obs_dim,
        action_dim,
        layers,
        heads,
        config_seq,
        config_rl,
        freeze_critic: bool,
        image_encoder_fn=lambda: None,
        gamma=0.99,
        tau=5e-3,
        clip=False,
        clip_grad_norm=1.0,
        device="cpu",
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.clip = False
        self.clip_grad_norm = clip_grad_norm

        self.freeze_critic = freeze_critic
        self.device = device

        self.algo = RL_ALGORITHMS[config_rl](
            action_dim=action_dim, automatic_entropy_tuning= None,target_entropy= None,entropy_alpha= None 
        )

        self.config_seq = get_config()
        self.config_seq.model.seq_model_config.n_layer = layers
        self.config_seq.model.seq_model_config.n_head = heads
        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        if image_encoder_fn() is None:
            observ_embedding_size = 64
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, self.config_seq["model"]["observ_embedder"]["hidden_size"], F.relu
            )
        else:  # for pixel observation, use external encoder
            self.observ_embedder = image_encoder_fn()
            observ_embedding_size = self.observ_embedder.embedding_size  # reset it

        self.action_embedder = utl.FeatureExtractor(
            action_dim, self.config_seq["model"]["action_embedder"]["hidden_size"], F.relu
        )
        self.reward_embedder = utl.FeatureExtractor(
            1, self.config_seq["model"]["reward_embedder"]["hidden_size"], F.relu
        )

        ## 2. build RNN model
        rnn_input_size = (
            self.config_seq["model"]["action_embedder"]["hidden_size"]
            + self.config_seq["model"]["reward_embedder"]["hidden_size"]
            + self.config_seq["model"]["observ_embedder"]["hidden_size"]
        )


        self.seq_model = GPT2(
            input_size=rnn_input_size, **(self.config_seq.to_dict()["model"]['seq_model_config']), max_seq_length=1000, device=self.device)
       

        ## 3. build actor-critic
        # q-value networks
        self.qf1, self.qf2 = self.algo.build_critic(
            input_size=self.seq_model.hidden_size + action_dim
            if self.algo.continuous_action
            else 128,
            # hidden_sizes=config_rl.config_critic.hidden_dims,
            action_dim=action_dim,
            hidden_sizes=[256,256],
        )
        # target networks
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)

        # policy network
        self.policy = self.algo.build_actor(
            input_size=self.seq_model.hidden_size,
            action_dim=self.action_dim,
            hidden_sizes=[256,256],
        )
        # target networks
        self.policy_target = deepcopy(self.policy)

        # use joint optimizer
        self.optimizer = Adam(self._get_parameters(), lr=0.0003)

    def _get_parameters(self):
        return [
            *self.observ_embedder.parameters(),
            *self.action_embedder.parameters(),
            *self.reward_embedder.parameters(),
            *self.seq_model.parameters(),
            *self.qf1.parameters(),
            *self.qf2.parameters(),
            *self.policy.parameters(),
        ]

    def get_hidden_states(
        self, prev_actions, rewards, observs, device, initial_internal_state=None
    ):
        """
        Get model's internal state
        """

        self.action_embedder = self.action_embedder.to(device)
        self.reward_embedder = self.reward_embedder.to(device)
        self.observ_embedder = self.observ_embedder.to(device)
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self.observ_embedder(observs)
        input_a = input_a.to(device)
        input_r = input_r.to(device)
        input_s = input_s.to(device)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)

        inputs = inputs.to(device)
        # feed into RNN: output (T+1, B, hidden_size)
        if initial_internal_state is None:  # training
            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1]
            )  
            self.seq_model.to(device)

            output, _ = self.seq_model(inputs, initial_internal_state)
            return output
        else:  # useful for one-step rollout during testing
            self.seq_model.to(device)
            output, current_internal_state = self.seq_model(
                inputs, initial_internal_state
            )
            return output, current_internal_state

    def reform_reward(self, score, next_score, g1, g2):
        """
        Basic reward function
        """
        base_reward = g1 * torch.tanh(score[:,2]-6)
        dynamic_reward = g2 * (next_score[:,2] - score[:,2])
        sofa_reward =  base_reward + dynamic_reward  
        sofa_reward = sofa_reward.unsqueeze(1)
        return sofa_reward

    def reform_reward_bias(self, score, next_score, g1, g2):
        """
        Reward function for biased score
        """
        base_reward = g1 * torch.tanh(score-6)
        dynamic_reward = g2 * (next_score- score)
        sofa_reward =  base_reward + dynamic_reward  
        return sofa_reward


    def forward(self, actions, rewards, observs, dones, masks, scores, next_scores, loss):
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
            == rewards.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with observs
        # return the hidden states (T+1, B, dim)
        # import time; t0 = time.time()
        hidden_states = self.get_hidden_states(
            prev_actions=actions, rewards=next_scores, observs=observs, device=self.device
        )

        hidden_states = hidden_states.to(self.device)

        ### 2. Critic loss

        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from target/current policy, (T+1, B, dim) including reaction to last obs
            # new_next_actions: (T+1, B, dim), new_next_log_probs: (T+1, B, 1 or A)
            self.policy = self.policy.to(self.device)
            self.policy_target = self.policy_target.to(self.device)
            new_next_actions, new_next_log_probs = self.algo.forward_actor_in_target(
                actor=self.policy,
                actor_target=self.policy_target,
                next_observ=hidden_states,
            )
            if self.algo.continuous_action:
                joint_q_embeds = torch.cat(
                    (hidden_states, new_next_actions), dim=-1
                )  # (T+1, B, dim)
            else:
                joint_q_embeds = hidden_states

            self.qf1_target = self.qf1_target.to(self.device)
            self.qf2_target = self.qf2_target.to(self.device)

            next_q1 = self.qf1_target(joint_q_embeds)
            next_q2 = self.qf2_target(joint_q_embeds)
            min_next_q_target = torch.min(next_q1, next_q2)

            # min_next_q_target (T+1, B, 1 or A)
            min_next_q_target += self.algo.entropy_bonus(new_next_log_probs)
            if not self.algo.continuous_action:
                min_next_q_target = (new_next_actions * min_next_q_target).sum(
                    dim=-1, keepdims=True
                )  # (T+1, B, 1)

            q_target = rewards + (1.0 - dones) * self.gamma * min_next_q_target
            q_target = q_target[1:]  # (T, B, 1)

        hidden_states = self.get_hidden_states(
            prev_actions=actions, rewards=scores, observs=observs, device=self.device
        )
        # Q(h(t), a(t)) (T, B, 1)
        # 3. joint embeds
        if self.algo.continuous_action:
            curr_joint_q_embeds = torch.cat(
                (hidden_states[:-1], actions[1:]), dim=-1
            )  # (T, B, dim)
        else:
            curr_joint_q_embeds = hidden_states[:-1]

        self.qf1 = self.qf1.to(self.device)
        self.qf2 = self.qf2.to(self.device)
        q1_pred = self.qf1(curr_joint_q_embeds)
        q2_pred = self.qf2(curr_joint_q_embeds)
        if not self.algo.continuous_action:
            stored_actions = actions[1:]  # (T, B, A)
            stored_actions = torch.argmax(
                stored_actions, dim=-1, keepdims=True
            )  # (T, B, 1)
            q1_pred = q1_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)
            q2_pred = q2_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks

        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        ### 3. Actor loss
        # Q(h(t), pi(h(t))) + H[pi(h(t))]
        # new_actions: (T+1, B, dim)
        new_actions, new_log_probs = self.algo.forward_actor(
            actor=self.policy, observ=hidden_states
        )

        if self.freeze_critic:
            ######## freeze critic parameters
            ######## and detach critic hidden states
            ######## such that the gradient only through new_actions
            if self.algo.continuous_action:
                new_joint_q_embeds = torch.cat(
                    (hidden_states.detach(), new_actions), dim=-1
                )  # (T+1, B, dim)
            else:
                new_joint_q_embeds = hidden_states.detach()

            new_joint_q_embeds = new_joint_q_embeds.to(self.device)
            freezed_qf1 = deepcopy(self.qf1).to(self.device)
            freezed_qf2 = deepcopy(self.qf2).to(self.device)
            q1 = freezed_qf1(new_joint_q_embeds)
            q2 = freezed_qf2(new_joint_q_embeds)

        else:
            if self.algo.continuous_action:
                new_joint_q_embeds = torch.cat(
                    (hidden_states, new_actions), dim=-1
                )  # (T+1, B, dim)
            else:
                new_joint_q_embeds = hidden_states

            q1 = self.qf1(new_joint_q_embeds)
            q2 = self.qf2(new_joint_q_embeds)

        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1 or A)


        policy_loss = -min_q_new_actions
        policy_loss += -self.algo.entropy_bonus(new_log_probs)

        if not self.algo.continuous_action:
            policy_loss = (new_actions * policy_loss).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)
            new_log_probs = (new_actions * new_log_probs).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)

        # ### 4. update
        policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
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
        
        total_loss = 0.5 * (qf1_loss + qf2_loss) + policy_loss

        outputs = {
            "critic_loss": (qf1_loss + qf2_loss).item(),
            "q1": (q1_pred.sum() / num_valid).item(),
            "q2": (q2_pred.sum() / num_valid).item(),
            "actor_loss": policy_loss.item(),
        }

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self._get_parameters(), self.clip_grad_norm
            )
            outputs["raw_grad_norm"] = grad_norm.item()

        self.optimizer.step()

        ### 5. soft update
        self.soft_target_update()

        ### 6. update others like alpha
        if new_log_probs is not None:
            # extract valid log_probs
            with torch.no_grad():
                current_log_probs = (new_log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)

    def report_grad_norm(self):
        return {
            "seq_grad_norm": utl.get_grad_norm(self.seq_model),
            "critic_grad_norm": utl.get_grad_norm(self.qf1),
            "actor_grad_norm": utl.get_grad_norm(self.policy),
        }

    def update(self, replay_buffer, loss, rew, rew_param):
        """
        Get episode for forward pass
        """
        # all are 3D tensor (T,B,dim)
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
            s = [1,1,1,1,1,1]
            if mod == "simple" or mod == "both":
                s = [0.8,0.8,0.8,0.8,0.8,0.8]

                if int(bias) < 1 or int(bias) > 6:
                  raise Exception("Invalid subscore")

                s[int(bias)- 1] = 2
                sofa1_param,sofa2_param,sofa3_param,sofa4_param,sofa5_param,sofa6_param = s[0], s[1], s[2], s[3], s[4], s[5]

            bias_scores = (sofa1_param*sofa1 + sofa2_param*sofa2 + sofa3_param*sofa3 + sofa4_param*sofa4 + sofa5_param*sofa5 + sofa6_param*sofa6)

            s = [1,1,1,1,1,1]
            if mod == "both":
                s = [0.8,0.8,0.8,0.8,0.8,0.8]
                if int(bias) < 1 or int(bias) > 6:
                  raise Exception("Invalid subscore")

                s[int(bias) - 1] = 2

            sofa1_param,sofa2_param,sofa3_param,sofa4_param,sofa5_param,sofa6_param = s[0], s[1], s[2], s[3], s[4], s[5]
            bias_next_scores = (sofa1_param*sofa1n + sofa2_param*sofa2n + sofa3_param*sofa3n + sofa4_param*sofa4n + sofa5_param*sofa5n + sofa6_param*sofa6n)
            reward = self.reform_reward_bias(bias_scores, bias_next_scores, g1, g2)



      
        inverted_done = 1.0 - not_done.to(self.device)

        # reshape data in to dimension (T+1, B, dim)
        actions, rewards, state, next_state, dones, scores, next_scores = action_c.unsqueeze(1), reward.unsqueeze(1), state.unsqueeze(1), next_state.unsqueeze(1), inverted_done.unsqueeze(1), ((scores[:,2]).reshape(-1,1).to(self.device)).unsqueeze(1), ((next_scores[:,2]).reshape(-1,1).to(self.device)).unsqueeze(1)
        next_actions = next_action_c.unsqueeze(1)
        actions = actions.to(self.device)
        next_actions = next_actions.to(self.device)
        rewards = rewards.to(self.device)

        scores, next_scores = scores.to(self.device), next_scores.to(self.device)
        observs = torch.cat((state[[0]], next_state), dim=0).to(self.device)  # (T+1, B, dim)
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



        return self.forward(actions, rewards, observs, dones, masks, scores, next_scores, loss)


    @torch.no_grad()
    def get_initial_info(self, max_attn_span: int = -1):
        """
        Create inital state
        """
        # here we assume batch_size = 1
        internal_state = self.seq_model.get_zero_internal_state()
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()
    
        return prev_action, reward, internal_state

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        device,
        deterministic=False,
    ):
        """
        Get model's actor next action
        """
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module
        prev_action = prev_action.unsqueeze(1)  # (1, B, dim)
        reward = reward.unsqueeze(1)  # (1, B, 1)
        obs = obs.unsqueeze(1)  # (1, B, 1)

        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action,
            rewards=reward,
            observs=obs,
            device=device,
            initial_internal_state=prev_internal_state,
        )
        hidden_state = hidden_state.to(device)

        self.policy = self.policy.to(device)
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        current_action = self.algo.select_action(
            actor=self.policy,
            observ=hidden_state,
            deterministic=deterministic,
        )

        return current_action, current_internal_state

    @torch.no_grad()
    def act_target(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        device,
        deterministic=False,
    ):
        """
        Get model's target actor next action
        """
        prev_action = prev_action.unsqueeze(1)  # (1, B, dim)
        reward = reward.unsqueeze(1)  # (1, B, 1)
        obs = obs.unsqueeze(1)  # (1, B, 1)

        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action,
            rewards=reward,
            observs=obs,
            device=device,
            initial_internal_state=prev_internal_state,
        )
        hidden_state = hidden_state.to(device)
        self.policy_target = self.policy_target.to(device)

        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        current_action = self.algo.select_action(
            actor=self.policy_target,
            observ=hidden_state,
            deterministic=deterministic,
        )

        return current_action, current_internal_state