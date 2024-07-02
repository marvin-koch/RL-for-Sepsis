"""
The classes and methods in this file are derived or pulled directly from https://github.com/sfujim/BCQ/tree/master/discrete_BCQ
which is a discrete implementation of BCQ by Scott Fujimoto, et al. and featured in the following 2019 DRL NeurIPS workshop paper:

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================


The code has been modified to train and evaluate an LSTM and Transformer TD3 implementation

"""

from scipy.stats import sem
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from rnn_models.policy_RNN import ModelFreeOffPolicy_Separate_RNN
from rnn_models.policy_Transformer_shared import ModelFreeOffPolicy_Shared_RNN

def run_LSTM_TD3(replay_buffer, test_replay_buffer, state_dim, action_dim,  device, parameters, writer, task, load, loss, reward, reward_param):
    """
    Run TD3 with LSTM encoder, we can either train or evaluate the model. There is also an option to run multiple models.
    
    """

    print("Run LSTM")
    buffer_dir = parameters['buffer_dir']
    test_buffer_dir = parameters['test_buffer_dir']


    policy = ModelFreeOffPolicy_Separate_RNN(state_dim, action_dim, "lstm", "td3", 16, 32, 16, 16, [256,256], [256,256], device, td3={"automatic_entropy_tuning": None,"target_entropy": None,"entropy_alpha": None })
    policy.to(device)
    

    replay_buffer.load(buffer_dir, bootstrap=True)
    test_replay_buffer.load(test_buffer_dir, bootstrap=True)


    training_iters = 0

    if task == 'train':
        print("train_lstm")
        
        #load model
        if not load == "none":
            l = load.split("_")
            policy = ModelFreeOffPolicy_Separate_RNN(state_dim, action_dim, "lstm", "td3", 16, 32, 16, int(l[-1][:-4]), [256,256], [256,256], device, td3={"automatic_entropy_tuning": None,"target_entropy": None,"entropy_alpha": None })

        #train
        while training_iters < parameters["max_timesteps"]:
            
            for _ in range(int(parameters["eval_freq"])):
                qf1_loss, qf2_loss, policy_loss = policy.update(replay_buffer, loss, reward, reward_param)

            training_iters += int(parameters["eval_freq"])
            print(f"Training iterations: {training_iters}")


        torch.save(policy.state_dict(), "trained_models/lstm_td3_default_16.pth" if load == "none" else load)


    elif task == 'eval':
        print("evaluate_lstm")

        #load model
        load_model = "trained_models/lstm_td3_16.pth" if load == "none" else load
        l = load_model.split("_")
        policy = ModelFreeOffPolicy_Separate_RNN(state_dim, action_dim, "lstm", "td3", 16, 32, 16, int(l[-1][:-4]), [256,256], [256,256], device, td3={"automatic_entropy_tuning": None,"target_entropy": None,"entropy_alpha": None })



        policy.load_state_dict(torch.load(load_model))           

        #perfrom evaluation 
        direct_eval_lstm(policy, test_replay_buffer,  [0, 1], device, parameters, action_dim, state_dim)
        lstm_plot_action_dist(policy, test_replay_buffer,  parameters, action_dim, device)
        rnn_plot_ucurve(policy, test_replay_buffer, device, parameters, action_dim)
        rnn_plot_action_sofa(policy, test_replay_buffer,  parameters, action_dim, device)

    elif task=='eval_multiple':
        print("evaluate multiple lstm")

        #load model
        models = ["trained_models/lstm_td3_16.pth", "trained_models/lstm_td3_42.pth","trained_models/lstm_td3_92.pth","trained_models/lstm_td3_128.pth"]
        load_model = models if load == "none" else load.split(",")
        l = load_model.split("_")
        policies = []
        for m in load_model:
            l = load_model.split("_")
            policy = ModelFreeOffPolicy_Separate_RNN(state_dim, action_dim, "lstm", "td3", 16, 32, 16, int(l[-1][:-4]), [256,256], [256,256], device, td3={"automatic_entropy_tuning": None,"target_entropy": None,"entropy_alpha": None })

            policy.load_state_dict(torch.load(m))
            policies.append(policy)

        #perfrom evaluation (only qualitative)
        rnn_mutiple_action_sofa(policies,test_replay_buffer, parameters, action_dim, device)
        rnn_multiple_plot_ucurve(policies, test_replay_buffer, device, parameters, action_dim)
    else:
        print("error")


def run_Transformer_TD3(replay_buffer, test_replay_buffer, state_dim, action_dim,  device, parameters, writer, task, load, loss, reward, reward_param):
    """
    Run TD3 with LSTM encoder, we can either train or evaluate the model. There is also an option to run multiple models.
    """

    print("Run Transformer")
    buffer_dir = parameters['buffer_dir']
    test_buffer_dir = parameters['test_buffer_dir']


    policy = ModelFreeOffPolicy_Shared_RNN(state_dim, action_dim, 1,1,"gpt", "td3",True, device=device)
    policy.to(device)
    

    replay_buffer.load(buffer_dir, bootstrap=True)
    test_replay_buffer.load(test_buffer_dir, bootstrap=True)

    training_iters = 0

    if task=='train':
        print("train_trans")
        if not load == "none":
            policy = ModelFreeOffPolicy_Shared_RNN(state_dim, action_dim, int(load[-7]), int(load[-5]),"gpt", "td3",True, device=device)


        while training_iters < parameters["max_timesteps"]:
            
            for _ in range(int(parameters["eval_freq"])):
                outputs = policy.update(replay_buffer, loss, reward, reward_param)

            training_iters += int(parameters["eval_freq"])
            print(f"Training iterations: {training_iters}")

        torch.save(policy.state_dict(), "trained_models/trans_td3_default_1_1.pth" if load == "none" else load)

    elif task=='eval':
        print("evaluate_trans")

        #load model
        load_model = "trained_models/trans_td3_1_1.pth" if load == "none" else load
        policy = ModelFreeOffPolicy_Shared_RNN(state_dim, action_dim, int(load_model[-7]), int(load_model[-5]),"gpt", "td3",True, device=device)
        policy.load_state_dict(torch.load(load_model))    

        #perfrom evaluation
        direct_eval_transformers(policy, test_replay_buffer, [0, 1], device, parameters, action_dim, state_dim)
        transformer_plot_action_dist(policy, test_replay_buffer,  parameters, action_dim, device)
        rnn_plot_ucurve(policy, test_replay_buffer, device, parameters, action_dim)
        rnn_plot_action_sofa(policy, test_replay_buffer, parameters, action_dim, device, isTransformer=True)

    elif task=='eval_multiple':
        print("evaluate multiple transformers")

        #load model
        models = ["trained_models/trans_td3_1_1.pth", "trained_models/trans_td3_2_2.pth", "trained_models/trans_td3_4_4.pth", "trained_models/trans_td3_2_4.pth"]
        load_model = models if load == "none" else load.split(",")
        policies = []
        for m in load_model:
            policy = ModelFreeOffPolicy_Shared_RNN(state_dim, action_dim, int(m[-7]), int(m[-5]),"gpt", "td3",True, device=device)
            policy.to(device)
            policy.load_state_dict(torch.load(m))
            policies.append(policy)

        #perfrom evaluation (only qualitative)
        rnn_mutiple_action_sofa(policies,test_replay_buffer, parameters, action_dim, device, isTransformer=True)
        rnn_multiple_plot_ucurve(policies, test_replay_buffer, device, parameters, action_dim)
    else:
        print("error")



def direct_eval_lstm(rl_policy, replay_buffer, vc_range, device, parameters, action_dim, state_dim):
    """
    Quantative evaluation with direct method of LSTM model
    rl_policy: the policy to be evaluated
    replay_buffer: test replay buffer with bootstrap == True
    iv_range and vc_range: the min and max actions for the 2 actions, used to create uniform random policy
    """
    eval_iters = 0
    Q_e, Q_c = 0, 0
    
    rl_policy.critic.eval()
    rl_policy.actor.eval()
    rl_policy.critic_target.eval()
    rl_policy.actor_target.eval()

    qs_rl = []
    qs_cl = []
    sofas = []

    while eval_iters < parameters["eval_steps"]:

        state, action_c,next_action_c,  next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        action, reward, internal_state = rl_policy.get_initial_info()
        
        batch_size = action_c.shape[0]
        action = action.to(device)
        reward = reward.to(device)
        action_c = action_c.to(device)
        scores = scores.to(device)
        state = state.to(device)
   

        batch_size = action_c.shape[0]
        for b in range(batch_size):
            action_c_, scores_, next_scores_, state_, next_state_  =  action_c[b].unsqueeze(0), scores[b].unsqueeze(0), next_scores[b].unsqueeze(0), state[b].unsqueeze(0), next_state[b].unsqueeze(0)
            (action_rl, _, _, _), internal_state = rl_policy.act(internal_state, action, reward, state_, device, True)

            action_, action_c__, reward_, state_, next_state_ = action.unsqueeze(1), action_c_.unsqueeze(1), reward.unsqueeze(1), state_.unsqueeze(1), next_state_.unsqueeze(1)
            action = action_c_

            reward = scores_[:,2].unsqueeze(1)


            action_rl = action_rl.unsqueeze(1)
       

            Q1_estimate, Q2_estimate = rl_policy.critic_target(action_, reward_, state_, action_rl)
            Q_estimate = torch.min(Q1_estimate, Q2_estimate).cpu()
            

            Q1_clinician, Q2_clinician = rl_policy.critic_target(action_, reward_, state_, action_c__)
            Q_clinician = torch.min(Q1_clinician, Q2_clinician).cpu()

            qs_rl.append(Q_estimate.sum().detach().cpu().numpy())
            qs_cl.append(Q_clinician.sum().detach().cpu().numpy())
            sofas.append(reward.cpu())

            Q_e += Q_estimate.sum()
            Q_c += Q_clinician.sum()
        
        eval_iters += 1
    

    sofas = np.concatenate(sofas, axis=None)
    qs_cl = np.concatenate(qs_cl, axis=None)
    qs_rl = np.concatenate(qs_rl, axis=None)
    
    xs, qclmean, qrlmean, qclse, qrlse = [], [], [], [], []
    for sofa in range(0, 25):
        sofa_idx = np.where(sofas==sofa)[0]
        xs.append(sofa)
        qclmean.append(qs_cl[sofa_idx].mean())
        qclse.append(sem(qs_cl[sofa_idx])) 
        qrlmean.append(qs_rl[sofa_idx].mean())
        qrlse.append(sem(qs_rl[sofa_idx])) 

    qclmean = np.array(qclmean)
    qclse = np.array(qclse)
    qrlmean = np.array(qrlmean)
    qrlse = np.array(qrlse)

    plt.figure()
    plt.subplot(121)
    plt.plot(xs,qclmean, '-o', color='mediumseagreen', label = 'Clinician policy')
    plt.fill_between(xs, qclmean - qclse, qclmean + qclse, color='mediumseagreen', alpha=0.5)
    plt.plot(xs,qrlmean, '-o', color='darkgreen', label = 'RL policy')
    plt.fill_between(xs, qrlmean - qrlse, qrlmean + qrlse, color='darkgreen', alpha=0.5)
    plt.xlabel("SOFA score")
    plt.ylabel("Mean Q-Value")
    plt.legend()
    plt.title('Mean Q-Value per SOFA Score')
    plt.show()

    res_e = (Q_e/(eval_iters*batch_size)).item()
    res_c = (Q_c/(eval_iters*batch_size)).item()
    print('Q estimate', Q_e/(eval_iters*batch_size))
    print('Q clinician', Q_c/(eval_iters*batch_size))

    import csv   
    fields=[res_e, res_c, 0, -1]
    with open('res.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

 
def direct_eval_transformers(rl_policy, replay_buffer,  vc_range, device, parameters, action_dim, state_dim):
    """
    Quantative evaluation with direct method of Transformer model
    rl_policy: the policy to be evaluated
    replay_buffer: test replay buffer with bootstrap == True
    iv_range and vc_range: the min and max actions for the 2 actions, used to create uniform random policy
    """
    eval_iters = 0
    Q_e, Q_c, Q_r, Q_z = 0, 0, 0, 0
    
    rl_policy.qf1.eval()
    rl_policy.qf2.eval()
    rl_policy.qf1_target.eval()
    rl_policy.qf2_target.eval()
    rl_policy.policy.eval()
    rl_policy.policy_target.eval()
    
    action, reward, internal_state = rl_policy.get_initial_info()
    while eval_iters < parameters["eval_steps"]:

        state, action_c,next_action_c,  next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        action, reward, internal_state = rl_policy.get_initial_info()
        

        batch_size = action_c.shape[0]
        action = action.to(device)
        reward = reward.to(device)
        action_c = action_c.to(device)
        scores = scores.to(device)
        state = state.to(device)
   

        batch_size = action_c.shape[0]
        for b in range(batch_size):
            action_c_, scores_, next_scores_, state_, next_state_  =  action_c[b].unsqueeze(0), scores[b].unsqueeze(0), next_scores[b].unsqueeze(0), state[b].unsqueeze(0), next_state[b].unsqueeze(0)
            (action_rl, _, _, _), internal_state_new = rl_policy.act(internal_state, action, reward, state_, device, True)
            action_, action_c__, reward_, state_, next_state_ = action.unsqueeze(1), action_c_.unsqueeze(1), reward.unsqueeze(1), state_.unsqueeze(1), next_state_.unsqueeze(1)
            action = action_c_
            reward = scores_[:,2].unsqueeze(1)


            action_rl = action_rl.unsqueeze(1)
            rl_policy = rl_policy.to(device)
            hidden_state, current_intern = rl_policy.get_hidden_states(action_, reward_, state_, device,internal_state)
            hidden_state = hidden_state.to(device)
            internal_state = internal_state_new


            joint_q_embeds = torch.cat(
                (hidden_state, action_rl), dim=-1
            )  # (T+1, B, dim)

            Q1_estimate, Q2_estimate = rl_policy.qf1_target(joint_q_embeds), rl_policy.qf2_target(joint_q_embeds)
            Q_estimate = torch.min(Q1_estimate, Q2_estimate).cpu()
            

            joint_q_embeds = torch.cat(
                (hidden_state, action_c__), dim=-1
            )  # (T+1, B, dim)


            Q1_clinician, Q2_clinician = rl_policy.qf1_target(joint_q_embeds), rl_policy.qf2_target(joint_q_embeds)
            Q_clinician = torch.min(Q1_clinician, Q2_clinician).cpu()

       
            Q_e += Q_estimate.sum()
        
            Q_c += Q_clinician.sum()
        
        eval_iters += 1
    
 
    res_e = (Q_e/(eval_iters*batch_size)).item()
    res_c = (Q_c/(eval_iters*batch_size)).item()
    print('Q estimate', Q_e/(eval_iters*batch_size))
    print('Q clinician', Q_c/(eval_iters*batch_size))
      
    import csv   
    fields=[res_e, res_c, 1, -2]
    with open('res.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
   
    
def lstm_plot_action_dist(rl_policy, replay_buffer,  parameters, action_dim, device):
    """
    Plot IV fluid and vasopressor distribution for LSTM model
    """
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    ax1 = plt.subplot(2, 1, 1)
    
    eval_iters = 0
    
    xs_rl = []
    xs_cl = []
    ys_rl = []
    ys_cl = []

    rl_policy.critic.eval()
    rl_policy.actor.eval()
    rl_policy.critic_target.eval()
    rl_policy.actor_target.eval()

    while eval_iters < parameters["eval_steps"]:
        action, reward, internal_state = rl_policy.get_initial_info()
        state, action_c, next_action_c, next_state, reward, scores,next_scores, outcome, done = replay_buffer.sample()
        batch_size = action_c.shape[0]

        action = action.to(device)
        action_c = action_c.to(device)
        reward = reward.to(device)
        next_scores = next_scores.to(device)
        state = state.to(device)
        next_state = next_state.to(device)
        reward = scores[0].unsqueeze(0)[:,2].unsqueeze(1)
        for b in range(batch_size):

            (action_rl, _, _, _), internal_state = rl_policy.act_target(internal_state, action, reward, state[b].unsqueeze(0), device, True)
            action = action_c[b].unsqueeze(0)
            reward = scores[b].unsqueeze(0)[:,2].unsqueeze(1)
            iv_rl = action_rl[:, 0]
            
            xs_rl.append(iv_rl.detach().cpu().numpy())
            xs_cl.append(action_c[b].unsqueeze(0)[:,0].cpu().numpy())

            vc_rl = action_rl[:, 1]
            ys_rl.append(vc_rl.detach().cpu().numpy())
            ys_cl.append(action_c[b].unsqueeze(0)[:,1].cpu().numpy())

            if done[b] == False:
               break 
            
        print(eval_iters)
        eval_iters+=1


    
    xs_rl = np.concatenate(xs_rl, axis=None)
    xs_cl = np.concatenate(xs_cl, axis=None)
    
    xs_cl = xs_cl * 2668
    xs_rl = xs_rl * 2668

    ys_rl = np.concatenate(ys_rl, axis=None)
    ys_rl = ys_rl * 1.187
    ys_cl = np.concatenate(ys_cl, axis=None)
    ys_cl = ys_cl * 1.187
    

    print('rl iv mean',xs_rl.mean())
    print('cl iv mean',xs_cl.mean())
    sns.set_style('whitegrid')
    ax = plt.subplot(2, 1, 1)
    sns.distplot(xs_cl, color='mediumseagreen', kde_kws={"clip":(0,2000)}, hist_kws={"range":(0,2000)}, ax = ax1)
    sns.distplot(xs_rl, color='darkgreen', kde_kws={"clip":(0,2000)}, hist_kws={"range":(0,2000)}, ax = ax1)

    ax.title.set_text('IV fluids dosage distribution')
    ax.title.set_size(20)

    ax.legend(['Clinician policy', 'RL policy'], fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_yscale('log')

    sns.distplot(ys_cl, color='skyblue', kde_kws={"clip":(0.0, 1.0)}, hist_kws={"range":(0.0,1.0)},ax = ax2)
    sns.distplot(ys_rl, color='royalblue', kde_kws={"clip":(0.0, 1.0)}, hist_kws={"range":(0.0,1.0)}, ax = ax2)
    
    ax2.title.set_text('Vasopressors dosage distribution')
    ax2.title.set_size(20)
    ax2.legend(['Clinician policy', 'RL policy'], fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    
    plt.show()
    
def transformer_plot_action_dist(rl_policy, replay_buffer,  parameters, action_dim, device):
    """
    Plot IV fluid and vasopressor distribution for Transformer model
    """
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    ax1 = plt.subplot(2, 1, 1)
    
    eval_iters = 0
    
    xs_rl = []
    xs_cl = []
    ys_rl = []
    ys_cl = []

    rl_policy.qf1.eval()
    rl_policy.qf2.eval()
    rl_policy.qf1_target.eval()
    rl_policy.qf2_target.eval()
    rl_policy.policy.eval()
    rl_policy.policy_target.eval()
    while eval_iters < parameters["eval_steps"]:
        action, reward, internal_state = rl_policy.get_initial_info()
        state, action_c, next_action_c, next_state, reward, scores,next_scores, outcome, done = replay_buffer.sample()
        batch_size = action_c.shape[0]

        
        action = action.to(device)
        action_c = action_c.to(device)
        reward = reward.to(device)
        next_scores = next_scores.to(device)
        state = state.to(device)
        next_state = next_state.to(device)
    
        reward = scores[0].unsqueeze(0)[:,2].unsqueeze(1)
        for b in range(batch_size):

       
            (action_rl, _, _, _), internal_state = rl_policy.act_target(internal_state, action, reward, state[b].unsqueeze(0), device, True)
            action = action_c[b].unsqueeze(0)
            reward = scores[b].unsqueeze(0)[:,2].unsqueeze(1)
            iv_rl = action_rl[:, 0]
            
            xs_rl.append(iv_rl.detach().cpu().numpy())
            xs_cl.append(action_c[b].unsqueeze(0)[:,0].cpu().numpy())

            vc_rl = action_rl[:, 1]
            ys_rl.append(vc_rl.detach().cpu().numpy())
            ys_cl.append(action_c[b].unsqueeze(0)[:,1].cpu().numpy())

            if done[b] == False:
               break 
            
        print(eval_iters)
        eval_iters+=1 
    action, reward, internal_state = rl_policy.get_initial_info()

    xs_rl = np.concatenate(xs_rl, axis=None)
    xs_cl = np.concatenate(xs_cl, axis=None)
    
    xs_cl = xs_cl * 2668
    xs_rl = xs_rl * 2668

    ys_rl = np.concatenate(ys_rl, axis=None)
    ys_rl = ys_rl * 1.187
    ys_cl = np.concatenate(ys_cl, axis=None)
    ys_cl = ys_cl * 1.187
    

    print('rl iv mean',xs_rl.mean())
    print('cl iv mean',xs_cl.mean())
    sns.set_style('whitegrid')
    ax = plt.subplot(2, 1, 1)
    sns.distplot(xs_cl, color='mediumseagreen', kde_kws={"clip":(0,2000)}, hist_kws={"range":(0,2000)}, ax = ax1)
    sns.distplot(xs_rl, color='darkgreen', kde_kws={"clip":(0,2000)}, hist_kws={"range":(0,2000)}, ax = ax1)


    ax.title.set_text('IV fluids dosage distribution')
    ax.title.set_size(20)

    ax.legend(['Clinician policy', 'RL policy'], fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_yscale('log')

    sns.distplot(ys_cl, color='skyblue', kde_kws={"clip":(0.0, 1.0)}, hist_kws={"range":(0.0,1.0)},ax = ax2)
    sns.distplot(ys_rl, color='royalblue', kde_kws={"clip":(0.0, 1.0)}, hist_kws={"range":(0.0,1.0)}, ax = ax2)
    
    ax2.title.set_text('Vasopressors dosage distribution')
    ax2.title.set_size(20)
    ax2.legend(['Clinician policy', 'RL policy'], fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    
    plt.show()
    
def rnn_plot_ucurve(rl_policy, replay_buffer, device, parameters, action_dim):
    """
    Plot U-curve
    """
    plt.figure(figsize=(14, 6)) 
    rnn_u_plot_iv(rl_policy, replay_buffer, device, parameters, action_dim)
    rnn_u_plot_vc(rl_policy, replay_buffer, device, parameters, action_dim)
    plt.show()

def rnn_multiple_plot_ucurve(policies, replay_buffer, device, parameters, action_dim):
    """
    Plot U-curve for multiple policies
    """
    plt.figure(figsize=(14, 6)) 
    rnn_mutiple_u_plot_iv(policies, replay_buffer, device, parameters, action_dim)
    rnn_mutiple_u_plot_vc(policies, replay_buffer, device, parameters, action_dim)
    plt.show()


def rnn_u_plot_vc(rl_policy, replay_buffer, device, parameters, action_dim):
    """
    Plot vasopressor U-curve (helper function)
    """
    plt.subplot(122)
    eval_iters = 0
    vc_diffs_rl = []
    vc_diffs_no = []
    death =[]

    while eval_iters < parameters["eval_steps"]:

        state, action_c, next_action_c, next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        action, reward, internal_state = rl_policy.get_initial_info()
        no_action = torch.zeros_like(next_action_c)
        outcome = torch.where(outcome == -1, 1, 0)
        batch_size = action_c.shape[0]

        action = action.to(device)
        reward = reward.to(device)


        scores = scores.to(device)
        state = state.to(device)
   
        for b in range(batch_size):
            (action_rl, _, _, _), internal_state = rl_policy.act(internal_state, action, reward, state[b].unsqueeze(0), device, True)
            action = action_c[b].unsqueeze(0)
            reward = (scores[b].unsqueeze(0))[:,2].unsqueeze(1)

            action_rl = action_rl * 1.187
            next_action_c = next_action_c * 1.187
            batch_size = action_c.shape[0]
            
            next_action_c_ = next_action_c[b].unsqueeze(0).to(device)
            no_action_ = no_action[b].unsqueeze(0).to(device)
            action_rl = action_rl.to(device)


            vc_r = action_rl[:,1] - next_action_c_[:,1]
            vc_n = no_action_[:, 1] - next_action_c_[:, 1]
            
            vc_diffs_rl.append(vc_r.cpu().detach().numpy())
            vc_diffs_no.append(vc_n.cpu().detach().numpy())

            death.append(outcome.cpu().detach().numpy())
        eval_iters+=1
    
    
    vc_diffs_rl = np.concatenate(vc_diffs_rl, axis = None)
    vc_diffs_no = np.concatenate(vc_diffs_no, axis = None)

    
    death = np.concatenate(death, axis = None)
    
        

    mort_vc_rl = []
    mort_vc_no = []
    bin_vc= []
    std_vc_rl = []
    std_vc_no = []
    i = 1.0
    while i >=-0.6:
        idx_vc_rl = np.where((vc_diffs_rl>i-0.05) & (vc_diffs_rl<i+0.05))[0]
        death_vc_rl = death[idx_vc_rl]

        idx_vc_no = np.where((vc_diffs_no>i-0.05) & (vc_diffs_no<i+0.05))[0]
        death_vc_no = death[idx_vc_no]

        
        death_mean_vc_rl = (death_vc_rl.sum())/len(death_vc_rl)
        death_mean_vc_no = (death_vc_no.sum())/len(death_vc_no)

        
        death_se_vc_rl = sem(death_vc_rl)
        death_se_vc_no = sem(death_vc_no)


        mort_vc_rl.append(death_mean_vc_rl)
        mort_vc_no.append(death_mean_vc_no)


        bin_vc.append(i)
        std_vc_rl.append(death_se_vc_rl)
        std_vc_no.append(death_se_vc_no)
        i-=0.1

    mort_vc_rl  = np.array(mort_vc_rl)
    mort_vc_no  = np.array(mort_vc_no)
    bin_vc = np.array(bin_vc)
    std_vc_rl = np.array(std_vc_rl)
    std_vc_no = np.array(std_vc_no)

    plt.plot(bin_vc, mort_vc_rl, color='skyblue', label = 'RL policy')
    plt.fill_between(bin_vc, mort_vc_rl - 1*std_vc_rl,  mort_vc_rl + 1*std_vc_rl, color='skyblue', alpha = 0.5)
    

    plt.plot(bin_vc, mort_vc_no, color='royalblue', label = 'Clinician policy')
    plt.fill_between(bin_vc, mort_vc_no - 1*std_vc_no,  mort_vc_no + 1*std_vc_no, color='royalblue', alpha = 0.5)

    plt.margins(x=0.1, y=0.2)  
    plt.xlabel("Recommend minus clinician dosage",fontsize=17)
    plt.ylabel("Observed mortality", fontsize=17)
    plt.legend(fontsize=20)
    plt.title('VC fluids', fontsize=17)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    
def rnn_mutiple_u_plot_vc(policies, replay_buffer, device, parameters, action_dim):
    """
    Plot vasopressor U-curve for multiple policies (helper function)
    """
    plt.subplot(122)
    eval_iters = 0
    vc_diffs_rl = []
    vc_diffs_rl2 = []
    vc_diffs_rl3 = []
    vc_diffs_rl4 = []
    vc_diffs_no = []
    death =[]

    while eval_iters < parameters["eval_steps"]:

        state, action_c, next_action_c, next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        action, reward, internal_state = policies[0].get_initial_info()
        action2, reward2, internal_state2 = policies[1].get_initial_info()
        action3, reward3, internal_state3 = policies[2].get_initial_info()
        action4, reward4, internal_state4 = policies[3].get_initial_info()
        no_action = torch.zeros_like(next_action_c)
        outcome = torch.where(outcome == -1, 1, 0)
        batch_size = action_c.shape[0]

        action = action.to(device)
        reward = reward.to(device)


        scores = scores.to(device)
        state = state.to(device)
   
        for b in range(batch_size):
            (action_rl, _, _, _), internal_state = policies[0].act(internal_state, action, reward, state[b].unsqueeze(0), device, True)
            (action_rl2, _, _, _), internal_state2 = policies[1].act(internal_state2, action, reward, state[b].unsqueeze(0), device, True)
            (action_rl3, _, _, _), internal_state3 = policies[2].act(internal_state3, action, reward, state[b].unsqueeze(0), device, True)
            (action_rl4, _, _, _), internal_state4 = policies[3].act(internal_state4, action, reward, state[b].unsqueeze(0), device, True)
            
            action = action_c[b].unsqueeze(0)
            reward = (scores[b].unsqueeze(0))[:,2].unsqueeze(1)

            action_rl = action_rl * 1.187
            action_rl2 = action_rl2 * 1.187
            action_rl3 = action_rl3 * 1.187
            action_rl4 = action_rl4 * 1.187
            next_action_c = next_action_c * 1.187
            batch_size = action_c.shape[0]
            
            next_action_c_ = next_action_c[b].unsqueeze(0).to(device)
            no_action_ = no_action[b].unsqueeze(0).to(device)
            action_rl = action_rl.to(device)
            action_rl2 = action_rl2.to(device)
            action_rl3 = action_rl3.to(device)
            action_rl4 = action_rl4.to(device)


            vc_r = action_rl[:,1] - next_action_c_[:,1]
            vc_r2 = action_rl2[:,1] - next_action_c_[:,1]
            vc_r3 = action_rl3[:,1] - next_action_c_[:,1]
            vc_r4 = action_rl3[:,1] - next_action_c_[:,1]
            vc_n = no_action_[:, 1] - next_action_c_[:, 1]
            
            vc_diffs_rl.append(vc_r.cpu().detach().numpy())
            vc_diffs_rl2.append(vc_r2.cpu().detach().numpy())
            vc_diffs_rl3.append(vc_r3.cpu().detach().numpy())
            vc_diffs_rl4.append(vc_r4.cpu().detach().numpy())
            vc_diffs_no.append(vc_n.cpu().detach().numpy())

            death.append(outcome.cpu().detach().numpy())
        eval_iters+=1
    
    
    vc_diffs_rl = np.concatenate(vc_diffs_rl, axis = None)
    vc_diffs_rl2 = np.concatenate(vc_diffs_rl2, axis = None)
    vc_diffs_rl3 = np.concatenate(vc_diffs_rl3, axis = None)
    vc_diffs_rl4 = np.concatenate(vc_diffs_rl4, axis = None)
    vc_diffs_no = np.concatenate(vc_diffs_no, axis = None)

    
    death = np.concatenate(death, axis = None)
    
        

    mort_vc_rl = []
    mort_vc_rl2 = []
    mort_vc_rl3 = []
    mort_vc_rl4 = []
    mort_vc_no = []
    bin_vc= []
    std_vc_rl = []
    std_vc_rl2 = []
    std_vc_rl3 = []
    std_vc_rl4 = []
    std_vc_no = []
    i = 1.0
    while i >=-0.6:
        idx_vc_rl = np.where((vc_diffs_rl>i-0.05) & (vc_diffs_rl<i+0.05))[0]
        death_vc_rl = death[idx_vc_rl]
        idx_vc_rl2 = np.where((vc_diffs_rl2>i-0.05) & (vc_diffs_rl2<i+0.05))[0]
        death_vc_rl2 = death[idx_vc_rl2]
        idx_vc_rl3 = np.where((vc_diffs_rl3>i-0.05) & (vc_diffs_rl3<i+0.05))[0]
        death_vc_rl3 = death[idx_vc_rl3]
        idx_vc_rl4 = np.where((vc_diffs_rl4>i-0.05) & (vc_diffs_rl4<i+0.05))[0]
        death_vc_rl4 = death[idx_vc_rl4]


        idx_vc_no = np.where((vc_diffs_no>i-0.05) & (vc_diffs_no<i+0.05))[0]
        death_vc_no = death[idx_vc_no]

        
        death_mean_vc_rl = (death_vc_rl.sum())/len(death_vc_rl)
        death_mean_vc_rl2 = (death_vc_rl2.sum())/len(death_vc_rl2)
        death_mean_vc_rl3 = (death_vc_rl3.sum())/len(death_vc_rl3)
        death_mean_vc_rl4 = (death_vc_rl4.sum())/len(death_vc_rl4)
        death_mean_vc_no = (death_vc_no.sum())/len(death_vc_no)

        
        death_se_vc_rl = sem(death_vc_rl)
        death_se_vc_rl2 = sem(death_vc_rl2)
        death_se_vc_rl3 = sem(death_vc_rl3)
        death_se_vc_rl4 = sem(death_vc_rl4)
        death_se_vc_no = sem(death_vc_no)


        mort_vc_rl.append(death_mean_vc_rl)
        mort_vc_rl2.append(death_mean_vc_rl2)
        mort_vc_rl3.append(death_mean_vc_rl3)
        mort_vc_rl4.append(death_mean_vc_rl4)
        mort_vc_no.append(death_mean_vc_no)


        bin_vc.append(i)
        std_vc_rl.append(death_se_vc_rl)
        std_vc_rl2.append(death_se_vc_rl2)
        std_vc_rl3.append(death_se_vc_rl3)
        std_vc_rl4.append(death_se_vc_rl4)
        std_vc_no.append(death_se_vc_no)
        i-=0.1

    mort_vc_rl  = np.array(mort_vc_rl)
    mort_vc_rl2  = np.array(mort_vc_rl2)
    mort_vc_rl3  = np.array(mort_vc_rl3)
    mort_vc_rl4  = np.array(mort_vc_rl4)
    mort_vc_no  = np.array(mort_vc_no)
    bin_vc = np.array(bin_vc)
    std_vc_rl = np.array(std_vc_rl)
    std_vc_rl2 = np.array(std_vc_rl2)
    std_vc_rl3 = np.array(std_vc_rl3)
    std_vc_rl4 = np.array(std_vc_rl4)
    std_vc_no = np.array(std_vc_no)

    plt.plot(bin_vc, mort_vc_rl, color='skyblue', label = '16 nodes')
    plt.fill_between(bin_vc, mort_vc_rl - 1*std_vc_rl,  mort_vc_rl + 1*std_vc_rl, color='skyblue', alpha = 0.5)

    plt.plot(bin_vc, mort_vc_rl2, color='green', label = '42 nodes')
    plt.fill_between(bin_vc, mort_vc_rl2 - 1*std_vc_rl2,  mort_vc_rl2 + 1*std_vc_rl2, color='green', alpha = 0.5)

    plt.plot(bin_vc, mort_vc_rl3, color='red', label = '92 nodes')
    plt.fill_between(bin_vc, mort_vc_rl3 - 1*std_vc_rl3,  mort_vc_rl3 + 1*std_vc_rl3, color='red', alpha = 0.5)

    plt.plot(bin_vc, mort_vc_rl4, color='purple', label = '128 nodes')
    plt.fill_between(bin_vc, mort_vc_rl4 - 1*std_vc_rl4,  mort_vc_rl4+ 1*std_vc_rl4, color='purple', alpha = 0.5)
   
    plt.plot(bin_vc, mort_vc_no, color='royalblue', label = 'Clinician policy')
    plt.fill_between(bin_vc, mort_vc_no - 1*std_vc_no,  mort_vc_no + 1*std_vc_no, color='royalblue', alpha = 0.5)

    plt.margins(x=0.1, y=0.2)  
    plt.xlabel("Recommend minus clinician dosage",fontsize=17)
    plt.ylabel("Observed mortality", fontsize=17)
    plt.legend(fontsize=20)
    plt.title('VC fluids', fontsize=17)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

   
   
    

def rnn_u_plot_iv(rl_policy, replay_buffer, device, parameters, action_dim):
    """
    Plot IV fluid U-curve (helper function)
    """
    plt.subplot(121)

    eval_iters = 0
    iv_diffs_rl, iv_diffs_no = [], []
    death =[]

    while eval_iters < parameters["eval_steps"]:

        state, action_c, next_action_c, next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        action, reward, internal_state = rl_policy.get_initial_info()
        outcome = torch.where(outcome == -1, 1, 0)
        batch_size = action_c.shape[0]

        action = action.to(device)
        reward = reward.to(device)
        state = state.to(device)
        next_action_c = next_action_c.to(device)
        no_action = torch.zeros_like(next_action_c)
        scores = scores.to(device)

   
        for b in range(batch_size):
            (action_rl, _, _, _), internal_state = rl_policy.act(internal_state, action, reward, state[b].unsqueeze(0), device, False)
            action = action_c[b].unsqueeze(0)
            reward = (scores[b].unsqueeze(0))[:,2].unsqueeze(1)

            action_rl = action_rl * 2668
            next_action_c_ = next_action_c[b].unsqueeze(0) * 2668

            next_action_c_ = next_action_c_.to(device)
            no_action_ = no_action[b].unsqueeze(0).to(device)
            action_rl = action_rl.to(device)

            batch_size = action_c.shape[0]
            iv_r = action_rl[:,0] - next_action_c_[:,0]
            iv_n = no_action_[:, 0] - next_action_c_[:, 0]
            
            iv_diffs_rl.append(iv_r.cpu().detach().numpy())
            iv_diffs_no.append(iv_n.cpu().detach().numpy())
        

            death.append(outcome.cpu().detach().numpy())
        eval_iters+=1
    
    iv_diffs_rl = np.concatenate(iv_diffs_rl, axis=None)
    iv_diffs_no = np.concatenate(iv_diffs_no, axis=None)

    death = np.concatenate(death, axis = None)
    

    mort_iv_no, mort_iv_rl = [],[]
    bin_iv = []
    std_iv_rl = []
    std_iv_no =  []
    i = 1200
    while i >=-1200:
        idx_iv_rl = np.where((iv_diffs_rl>i-100) & (iv_diffs_rl<i+100))[0]
        death_iv_rl = death[idx_iv_rl]

        idx_iv_no = np.where((iv_diffs_no>i-100) & (iv_diffs_no<i+100))[0]
        death_iv_no = death[idx_iv_no]
        
        death_mean_iv_rl = (death_iv_rl.sum())/len(death_iv_rl)
        death_mean_iv_no = (death_iv_no.sum())/len(death_iv_no)
        
        death_se_iv_rl = sem(death_iv_rl)
        death_se_iv_no = sem(death_iv_no)

        mort_iv_rl.append(death_mean_iv_rl)
        mort_iv_no.append(death_mean_iv_no)

        bin_iv.append(i)
        std_iv_rl.append(death_se_iv_rl)

        std_iv_no.append(death_se_iv_no)
        i-=100

    mort_iv_rl  = np.array(mort_iv_rl)
    mort_iv_no  = np.array(mort_iv_no)

    bin_iv = np.array(bin_iv)
    std_iv_rl = np.array(std_iv_rl)
    std_iv_no = np.array(std_iv_no)

    plt.plot(bin_iv, mort_iv_rl, color='mediumseagreen', label = 'RL policy')
    plt.fill_between(bin_iv, mort_iv_rl - 1*std_iv_rl,  mort_iv_rl + 1*std_iv_rl, color='mediumseagreen', alpha = 0.5)
    

    plt.plot(bin_iv, mort_iv_no, color='darkgreen', label = 'Clinician policy')
    plt.fill_between(bin_iv, mort_iv_no - 1*std_iv_no,  mort_iv_no + 1*std_iv_no, color='darkgreen', alpha = 0.5)

    plt.margins(x=0.1, y=0.2) 
    plt.xlabel("Recommend minus clinician dosage", fontsize=20)
    plt.ylabel("Observed mortality", fontsize=20)
    plt.legend(fontsize=20)
    plt.title('IV fluids', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)



    
def rnn_mutiple_u_plot_iv(policies, replay_buffer, device, parameters, action_dim):
    """
    Plot IV fluid U-curve for multiple policies (helper function)
    """
    plt.subplot(121)
    eval_iters = 0
    iv_diffs_rl, iv_diffs_no = [], []
    iv_diffs_rl2, iv_diffs_rl3 = [], []
    iv_diffs_rl4 = []
    death =[]

    while eval_iters < parameters["eval_steps"]:

        state, action_c, next_action_c, next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        action, reward, internal_state = policies[0].get_initial_info()
        action2, reward2, internal_state2 = policies[1].get_initial_info()
        action3, reward3, internal_state3 = policies[2].get_initial_info()
        action4, reward4, internal_state4 = policies[3].get_initial_info()
        outcome = torch.where(outcome == -1, 1, 0)
        batch_size = action_c.shape[0]

        action = action.to(device)
        action2 = action2.to(device)
        action3 = action3.to(device)
        action4 = action4.to(device)
        reward = reward.to(device)
        state = state.to(device)
        next_action_c = next_action_c.to(device)
        no_action = torch.zeros_like(next_action_c)
        scores = scores.to(device)


        for b in range(batch_size):

            (action_rl, _, _, _), internal_state = policies[0].act(internal_state, action, reward, state[b].unsqueeze(0), device, True)
            (action_rl2, _, _, _), internal_state2 = policies[1].act(internal_state2, action, reward, state[b].unsqueeze(0), device, True)
            (action_rl3, _, _, _), internal_state3 = policies[2].act(internal_state3, action, reward, state[b].unsqueeze(0), device, True)
            (action_rl4, _, _, _), internal_state4 = policies[3].act(internal_state4, action, reward, state[b].unsqueeze(0), device, True)
            action = action_c[b].unsqueeze(0)
            reward = (scores[b].unsqueeze(0))[:,2].unsqueeze(1)

            action_rl = action_rl * 2668
            action_rl2 = action_rl2 * 2668
            action_rl3 = action_rl3 * 2668
            action_rl4 = action_rl4 * 2668
            next_action_c_ = next_action_c[b].unsqueeze(0) * 2668

            next_action_c_ = next_action_c_.to(device)
            no_action_ = no_action[b].unsqueeze(0).to(device)
            action_rl = action_rl.to(device)
            action_rl2 = action_rl2.to(device)
            action_rl3 = action_rl3.to(device)
            action_rl4 = action_rl4.to(device)

            batch_size = action_c.shape[0]
            iv_r = action_rl[:,0] - next_action_c_[:,0]
            iv_r2 = action_rl2[:,0] - next_action_c_[:,0]
            iv_r3 = action_rl3[:,0] - next_action_c_[:,0]
            iv_r4 = action_rl4[:,0] - next_action_c_[:,0]
            iv_n = no_action_[:, 0] - next_action_c_[:, 0]
            
            iv_diffs_rl.append(iv_r.cpu().detach().numpy())
            iv_diffs_rl2.append(iv_r2.cpu().detach().numpy())
            iv_diffs_rl3.append(iv_r3.cpu().detach().numpy())
            iv_diffs_rl4.append(iv_r4.cpu().detach().numpy())
            iv_diffs_no.append(iv_n.cpu().detach().numpy())
        

            death.append(outcome.cpu().detach().numpy())
        eval_iters+=1
    
    iv_diffs_rl = np.concatenate(iv_diffs_rl, axis=None)
    iv_diffs_rl2 = np.concatenate(iv_diffs_rl2, axis=None)
    iv_diffs_rl3 = np.concatenate(iv_diffs_rl3, axis=None)
    iv_diffs_rl4 = np.concatenate(iv_diffs_rl4, axis=None)
    iv_diffs_no = np.concatenate(iv_diffs_no, axis=None)

    death = np.concatenate(death, axis = None)
    

    mort_iv_no, mort_iv_rl = [],[]
    mort_iv_rl2, mort_iv_rl3 = [], []
    mort_iv_rl4 = []
    bin_iv = []
    std_iv_rl = []
    std_iv_rl2 = []
    std_iv_rl3 = []
    std_iv_rl4 = []
    std_iv_no =  []
    i = 1200
    while i >=-1200:
        idx_iv_rl = np.where((iv_diffs_rl>i-100) & (iv_diffs_rl<i+100))[0]
        death_iv_rl = death[idx_iv_rl]

        idx_iv_rl2 = np.where((iv_diffs_rl2>i-100) & (iv_diffs_rl2<i+100))[0]
        death_iv_rl2 = death[idx_iv_rl2]

        idx_iv_rl3 = np.where((iv_diffs_rl3>i-100) & (iv_diffs_rl3<i+100))[0]
        death_iv_rl3 = death[idx_iv_rl3]

        idx_iv_rl4 = np.where((iv_diffs_rl4>i-100) & (iv_diffs_rl4<i+100))[0]
        death_iv_rl4 = death[idx_iv_rl4]

        idx_iv_no = np.where((iv_diffs_no>i-100) & (iv_diffs_no<i+100))[0]
        death_iv_no = death[idx_iv_no]
        
        death_mean_iv_rl = (death_iv_rl.sum())/len(death_iv_rl)
        death_mean_iv_rl2 = (death_iv_rl2.sum())/len(death_iv_rl2)
        death_mean_iv_rl3 = (death_iv_rl3.sum())/len(death_iv_rl3)
        death_mean_iv_rl4 = (death_iv_rl4.sum())/len(death_iv_rl4)
        death_mean_iv_no = (death_iv_no.sum())/len(death_iv_no)
        
        death_se_iv_rl = sem(death_iv_rl)
        death_se_iv_rl2 = sem(death_iv_rl2)
        death_se_iv_rl3 = sem(death_iv_rl3)
        death_se_iv_rl4 = sem(death_iv_rl4)
        death_se_iv_no = sem(death_iv_no)

        mort_iv_rl.append(death_mean_iv_rl)
        mort_iv_rl2.append(death_mean_iv_rl2)
        mort_iv_rl3.append(death_mean_iv_rl3)
        mort_iv_rl4.append(death_mean_iv_rl4)
        mort_iv_no.append(death_mean_iv_no)

        bin_iv.append(i)
        std_iv_rl.append(death_se_iv_rl)
        std_iv_rl2.append(death_se_iv_rl2)
        std_iv_rl3.append(death_se_iv_rl3)
        std_iv_rl4.append(death_se_iv_rl4)

        std_iv_no.append(death_se_iv_no)
        i-=100

    mort_iv_rl  = np.array(mort_iv_rl)
    mort_iv_rl2  = np.array(mort_iv_rl2)
    mort_iv_rl3  = np.array(mort_iv_rl3)
    mort_iv_rl4  = np.array(mort_iv_rl4)
    mort_iv_no  = np.array(mort_iv_no)

    bin_iv = np.array(bin_iv)
    std_iv_rl = np.array(std_iv_rl)
    std_iv_rl2 = np.array(std_iv_rl2)
    std_iv_rl3 = np.array(std_iv_rl3)
    std_iv_no = np.array(std_iv_no)

    plt.plot(bin_iv, mort_iv_rl, color='blue', label='16 nodes')
    plt.fill_between(bin_iv, mort_iv_rl - 1*std_iv_rl, mort_iv_rl + 1*std_iv_rl, color='blue', alpha=0.5)

    plt.plot(bin_iv, mort_iv_rl2, color='lightgreen', label='42 nodes')
    plt.fill_between(bin_iv, mort_iv_rl2 - 1*std_iv_rl2, mort_iv_rl2 + 1*std_iv_rl2, color='lightgreen', alpha=0.5)

    plt.plot(bin_iv, mort_iv_rl3, color='red', label='92 nodes')
    plt.fill_between(bin_iv, mort_iv_rl3 - 1*std_iv_rl3, mort_iv_rl3 + 1*std_iv_rl3, color='red', alpha=0.5)

    plt.plot(bin_iv, mort_iv_rl4, color='purple', label='128 nodes')
    plt.fill_between(bin_iv, mort_iv_rl4 - 1*std_iv_rl4, mort_iv_rl4 + 1*std_iv_rl4, color='purple', alpha=0.5)

    plt.plot(bin_iv, mort_iv_no, color='darkgreen', label='Clinician policy')
    plt.fill_between(bin_iv, mort_iv_no - 1*std_iv_no, mort_iv_no + 1*std_iv_no, color='darkgreen', alpha=0.5)

    plt.margins(x=0.1, y=0.2)  
    plt.xlabel("Recommend minus clinician dosage", fontsize=20)
    plt.ylabel("Observed mortality", fontsize=20)
    plt.legend(fontsize=20)
    plt.title('IV fluids', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

def rnn_plot_action_sofa(rl_policy, replay_buffer,  parameters, action_dim, device, isTransformer=False):
    """
    Plot average IV fluid and vasopressor dosage per SOFA score and subscores
    """
    eval_iters = 0
    ivs_rl = []
    ivs_cl = []
    vcs_rl = []
    vcs_cl = []

    if not isTransformer:
        rl_policy.critic.eval()
        rl_policy.actor.eval()
        rl_policy.critic_target.eval()
        rl_policy.actor_target.eval()
    else:
        rl_policy.qf1.eval()
        rl_policy.qf2.eval()
        rl_policy.qf1_target.eval()
        rl_policy.qf2_target.eval()
        rl_policy.policy.eval()
        rl_policy.policy_target.eval()
    
    
        
    sofas = []
    sofas1 = []
    sofas2 = []
    sofas3 = []
    sofas4 = []
    sofas5 = []
    sofas6 = []

    while eval_iters < parameters["eval_steps"]:

        state, action_c, next_action_c, next_state, reward, scores, sofa1, sofa2, sofa3, sofa4, sofa5, sofa6,n1,n2,n3,n4,n5,n6, next_scores,outcome, done = replay_buffer.sample(extra=True)
        action, reward, internal_state = rl_policy.get_initial_info()

        batch_size = action_c.shape[0]
        action = action.to(device)
        reward = reward.to(device)
        action_c = action_c.to(device)
        scores = scores.to(device)
        state = state.to(device)
   
        for b in range(batch_size):
            (action_rl, _, _, _), internal_state = rl_policy.act(internal_state, action, reward, state[b].unsqueeze(0), device, True)
            action = action_c[b].unsqueeze(0)
            reward = (scores[b].unsqueeze(0))[:,2].unsqueeze(1)
            action_c_ = action

            iv_rl = action_rl[:, 0]
            vc_rl = action_rl[:, 1]
            ivs_rl.append(iv_rl.detach().cpu().numpy())
            ivs_cl.append(action_c_[:,0].cpu().numpy())
            vcs_rl.append(vc_rl.detach().cpu().numpy())
            vcs_cl.append(action_c_[:,1].cpu().numpy())

            sofas.append(reward.cpu().numpy())
            sofas1.append(sofa1[b].unsqueeze(0).cpu().numpy())
            sofas2.append(sofa2[b].unsqueeze(0).cpu().numpy())
            sofas3.append(sofa3[b].unsqueeze(0).cpu().numpy())
            sofas4.append(sofa4[b].unsqueeze(0).cpu().numpy())
            sofas5.append(sofa5[b].unsqueeze(0).cpu().numpy())
            sofas6.append(sofa6[b].unsqueeze(0).cpu().numpy())
        

        eval_iters+=1

    
    ivs_rl = np.concatenate(ivs_rl, axis=None)
    vcs_rl = np.concatenate(vcs_rl, axis=None)
    sofas = np.concatenate(sofas, axis=None)
    sofas1 = np.concatenate(sofas1, axis=None)
    sofas2 = np.concatenate(sofas2, axis=None)
    sofas3 = np.concatenate(sofas3, axis=None)
    sofas4 = np.concatenate(sofas4, axis=None)
    sofas5 = np.concatenate(sofas5, axis=None)
    sofas6 = np.concatenate(sofas6, axis=None)
    ivs_cl = np.concatenate(ivs_cl, axis=None)
    vcs_cl = np.concatenate(vcs_cl, axis=None)

    ivs_rl = ivs_rl * 2668
    ivs_cl = ivs_cl * 2668
    vcs_rl = vcs_rl * 1.187
    vcs_cl = vcs_cl * 1.187

    num = 0
    for i in [sofas,sofas1,sofas2,sofas3,sofas4,sofas5,sofas6]:
        xs, vcrl_mean, vcrl_se, vccl_mean, vccl_se, ivcl_mean, ivcl_se, ivrl_mean, ivrl_se = [], [], [], [], [], [], [], [] ,[] 
        for sofa in range(0, 25):
            sofa_idx = np.where(i==sofa)[0]
            xs.append(sofa)
            vccl_mean.append(vcs_cl[sofa_idx].mean())
            vccl_se.append(sem(vcs_cl[sofa_idx])) 
            vcrl_mean.append(vcs_rl[sofa_idx].mean())
            vcrl_se.append(sem(vcs_rl[sofa_idx])) 
            ivcl_mean.append(ivs_cl[sofa_idx].mean())
            ivcl_se.append(sem(ivs_cl[sofa_idx])) 
            ivrl_mean.append(ivs_rl[sofa_idx].mean())
            ivrl_se.append(sem(ivs_rl[sofa_idx])) 

        vccl_mean = np.array(vccl_mean)
        vccl_se = np.array(vccl_se)
        vcrl_mean = np.array(vcrl_mean)
        vcrl_se = np.array(vcrl_se)
        ivcl_mean = np.array(ivcl_mean)
        ivcl_se = np.array(ivcl_se)
        ivrl_mean = np.array(ivrl_mean)
        ivrl_se = np.array(ivrl_se)


        plt.figure(figsize=(14, 6))  # Increase the figure size for better spacing
        plt.subplot(121)
        plt.plot(xs,ivcl_mean, '-o', color='mediumseagreen', label = 'Clinician policy')
        plt.fill_between(xs, ivcl_mean - ivcl_se, ivcl_mean + ivcl_se, color='mediumseagreen', alpha=0.5)
        plt.plot(xs,ivrl_mean, '-o', color='darkgreen', label = 'RL policy')
        plt.fill_between(xs, ivrl_mean - ivrl_se, ivrl_mean + ivrl_se, color='darkgreen', alpha=0.5)

        plt.margins(x=0.1, y=0.2)  # Increase margins to prevent overlap

       
        plt.xlabel("SOFA score" + ("" if num == 0 else " organ " + str(num)), fontsize=17)
        plt.ylabel("Mean Dosage", fontsize=17)
        plt.legend(fontsize=17)
        plt.title('IV fluids', fontsize=17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)



        plt.subplot(122)
        plt.plot(xs,vccl_mean, '-o', color='skyblue', label = 'Clinician policy')
        plt.fill_between(xs, vccl_mean - vccl_se, vccl_mean + vccl_se, color='skyblue',alpha=0.5)
        plt.plot(xs,vcrl_mean, '-o', color='royalblue', label = 'RL policy')
        plt.fill_between(xs, vcrl_mean - vcrl_se, vcrl_mean + vcrl_se, color='royalblue', alpha=0.5)

        plt.margins(x=0.1, y=0.2)  # Increase margins to prevent overlap
        plt.xlabel("SOFA score" + ("" if num == 0 else " organ " + str(num)), fontsize=17)
        plt.ylabel("Mean Dosage", fontsize=17)
        plt.legend(fontsize=17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.title('Vasopressors', fontsize=17)
        plt.show()
 
        num += 1

def rnn_mutiple_action_sofa(policies, replay_buffer,  parameters, action_dim, device, isTransformer=False):
    """
    Plot average IV fluid and vasopressor dosage per SOFA score and subscores for multiple policies
    """
    eval_iters = 0
    ivs_rl = []
    ivs_cl = []
    vcs_rl = []
    vcs_cl = []

    ivs_rl2 = []
    vcs_rl2 = []

    ivs_rl3 = []
    vcs_rl3 = []

    ivs_rl4 = []
    vcs_rl4 = []


    for rl_policy in policies:
        if not isTransformer:
            rl_policy.critic.eval()
            rl_policy.actor.eval()
            rl_policy.critic_target.eval()
            rl_policy.actor_target.eval()
        else:
            rl_policy.qf1.eval()
            rl_policy.qf2.eval()
            rl_policy.qf1_target.eval()
            rl_policy.qf2_target.eval()
            rl_policy.policy.eval()
            rl_policy.policy_target.eval()
    
    
        
    sofas = []
    sofas1 = []
    sofas2 = []
    sofas3 = []
    sofas4 = []

    while eval_iters < parameters["eval_steps"]:

        state, action_c, next_action_c, next_state, reward, scores, sofa1, sofa2, sofa3, sofa4, sofa5, sofa6,n1,n2,n3,n4,n5,n6, next_scores,outcome, done = replay_buffer.sample(extra=True)

        action, reward, internal_state = policies[0].get_initial_info()
        action2, reward2, internal_state2 = policies[1].get_initial_info()
        action3, reward3, internal_state3 = policies[2].get_initial_info()
        action4, reward4, internal_state4 = policies[3].get_initial_info()

        batch_size = action_c.shape[0]
        action = action.to(device)
        reward = reward.to(device)
        action2 = action2.to(device)
        reward2 = reward2.to(device)
        action3 = action3.to(device)
        reward3 = reward3.to(device)
        action4 = action4.to(device)
        reward4 = reward4.to(device)

        action_c = action_c.to(device)
        scores = scores.to(device)
        state = state.to(device)
   


        for b in range(batch_size):
            (action_rl, _, _, _), internal_state = policies[0].act(internal_state, action, reward, state[b].unsqueeze(0), device, True)
            (action_rl2, _, _, _), internal_state2 = policies[1].act(internal_state2, action2, reward2, state[b].unsqueeze(0), device, True)
            (action_rl3, _, _, _), internal_state3 = policies[2].act(internal_state3, action3, reward3, state[b].unsqueeze(0), device, True)
            (action_rl4, _, _, _), internal_state4 = policies[3].act(internal_state4, action4, reward4, state[b].unsqueeze(0), device, True)
            action = action_c[b].unsqueeze(0)
            reward = (scores[b].unsqueeze(0))[:,2].unsqueeze(1)
            action_c_ = action


            iv_rl = action_rl[:, 0]
            vc_rl = action_rl[:, 1]
            iv_rl2 = action_rl2[:, 0]
            vc_rl2 = action_rl2[:, 1]
            iv_rl3 = action_rl3[:, 0]
            vc_rl3 = action_rl3[:, 1]
            iv_rl4 = action_rl4[:, 0]
            vc_rl4 = action_rl4[:, 1]
   


            ivs_rl.append(iv_rl.detach().cpu().numpy())
            ivs_cl.append(action_c_[:,0].cpu().numpy())
            vcs_rl.append(vc_rl.detach().cpu().numpy())
            vcs_cl.append(action_c_[:,1].cpu().numpy())
            ivs_rl2.append(iv_rl2.detach().cpu().numpy())
            vcs_rl2.append(vc_rl2.detach().cpu().numpy())
            ivs_rl3.append(iv_rl3.detach().cpu().numpy())
            vcs_rl3.append(vc_rl3.detach().cpu().numpy())
            ivs_rl4.append(iv_rl4.detach().cpu().numpy())
            vcs_rl4.append(vc_rl4.detach().cpu().numpy())

            sofas.append(reward.cpu().numpy())
            sofas1.append(sofa1[b].unsqueeze(0).cpu().numpy())
            sofas2.append(sofa2[b].unsqueeze(0).cpu().numpy())
            sofas3.append(sofa3[b].unsqueeze(0).cpu().numpy())
            sofas4.append(sofa4[b].unsqueeze(0).cpu().numpy())

        eval_iters+=1

    
    ivs_rl = np.concatenate(ivs_rl, axis=None)
    vcs_rl = np.concatenate(vcs_rl, axis=None)
    ivs_rl2 = np.concatenate(ivs_rl2, axis=None)
    vcs_rl2 = np.concatenate(vcs_rl2, axis=None)
    ivs_rl3 = np.concatenate(ivs_rl3, axis=None)
    vcs_rl3 = np.concatenate(vcs_rl3, axis=None)
    ivs_rl4 = np.concatenate(ivs_rl4, axis=None)
    vcs_rl4 = np.concatenate(vcs_rl4, axis=None)

    sofas = np.concatenate(sofas, axis=None)
    sofas1 = np.concatenate(sofas1, axis=None)
    sofas2 = np.concatenate(sofas2, axis=None)
    sofas3 = np.concatenate(sofas3, axis=None)
    sofas4 = np.concatenate(sofas4, axis=None)

    ivs_cl = np.concatenate(ivs_cl, axis=None)
    vcs_cl = np.concatenate(vcs_cl, axis=None)

    ivs_rl = ivs_rl * 2668
    ivs_rl2 = ivs_rl2 * 2668
    ivs_rl3 = ivs_rl3 * 2668
    ivs_rl4 = ivs_rl4 * 2668

    ivs_cl = ivs_cl * 2668
    vcs_rl = vcs_rl * 1.187
    vcs_rl2 = vcs_rl2 * 1.187
    vcs_rl3 = vcs_rl3 * 1.187
    vcs_rl4 = vcs_rl4 * 1.187

    vcs_cl = vcs_cl * 1.187
    
    num = 0
    for i in [sofas,sofas1,sofas2,sofas3,sofas4]:
        xs, vcrl_mean, vcrl_se, vccl_mean, vccl_se, ivcl_mean, ivcl_se, ivrl_mean, ivrl_se = [], [], [], [], [], [], [], [] ,[] 
        vcrl_mean2, vcrl_se2, ivrl_mean2, ivrl_se2 = [],[],[],[]
        vcrl_mean3, vcrl_se3, ivrl_mean3, ivrl_se3 = [],[],[],[]
        vcrl_mean4, vcrl_se4, ivrl_mean4, ivrl_se4 = [],[],[],[]

        for sofa in range(0, 25):
            sofa_idx = np.where(i==sofa)[0]
            xs.append(sofa)
            vccl_mean.append(vcs_cl[sofa_idx].mean())
            vccl_se.append(sem(vcs_cl[sofa_idx])) 
            vcrl_mean.append(vcs_rl[sofa_idx].mean())
            vcrl_se.append(sem(vcs_rl[sofa_idx])) 

            vcrl_mean2.append(vcs_rl2[sofa_idx].mean())
            vcrl_se2.append(sem(vcs_rl2[sofa_idx])) 
            vcrl_mean3.append(vcs_rl3[sofa_idx].mean())
            vcrl_se3.append(sem(vcs_rl3[sofa_idx])) 
            vcrl_mean4.append(vcs_rl4[sofa_idx].mean())
            vcrl_se4.append(sem(vcs_rl4[sofa_idx])) 
 
            ivcl_mean.append(ivs_cl[sofa_idx].mean())
            ivcl_se.append(sem(ivs_cl[sofa_idx])) 
            ivrl_mean.append(ivs_rl[sofa_idx].mean())
            ivrl_se.append(sem(ivs_rl[sofa_idx])) 

            ivrl_mean2.append(ivs_rl2[sofa_idx].mean())
            ivrl_se2.append(sem(ivs_rl2[sofa_idx])) 
            ivrl_mean3.append(ivs_rl3[sofa_idx].mean())
            ivrl_se3.append(sem(ivs_rl3[sofa_idx])) 
            ivrl_mean4.append(ivs_rl4[sofa_idx].mean())
            ivrl_se4.append(sem(ivs_rl4[sofa_idx])) 
  
        vccl_mean = np.array(vccl_mean)
        vccl_se = np.array(vccl_se)
        vcrl_mean = np.array(vcrl_mean)
        vcrl_se = np.array(vcrl_se)
        vcrl_mean2 = np.array(vcrl_mean2)
        vcrl_se2 = np.array(vcrl_se2)
        vcrl_mean3 = np.array(vcrl_mean3)
        vcrl_se3 = np.array(vcrl_se3)
        vcrl_mean4 = np.array(vcrl_mean4)
        vcrl_se4 = np.array(vcrl_se4)

        ivcl_mean = np.array(ivcl_mean)
        ivcl_se = np.array(ivcl_se)

        ivrl_mean = np.array(ivrl_mean)
        ivrl_se = np.array(ivrl_se)

        ivrl_mean2 = np.array(ivrl_mean2)
        ivrl_se2 = np.array(ivrl_se2)

        ivrl_mean3 = np.array(ivrl_mean3)
        ivrl_se3 = np.array(ivrl_se3)

        ivrl_mean4 = np.array(ivrl_mean4)
        ivrl_se4 = np.array(ivrl_se4)


        plt.figure(figsize=(14, 6)) 

        plt.subplot(121)
        plt.plot(xs,ivcl_mean, '-o', color='mediumseagreen', label = 'Clinician policy')
        plt.fill_between(xs, ivcl_mean - ivcl_se, ivcl_mean + ivcl_se, color='mediumseagreen', alpha=0.5)

        plt.plot(xs,ivrl_mean, '-o', color='darkgreen', label = '16 nodes')

        plt.fill_between(xs, ivrl_mean - ivrl_se, ivrl_mean + ivrl_se, color='darkgreen', alpha=0.5)

        plt.plot(xs,ivrl_mean2, '-o', color='lime', label = '42 nodes')
        plt.fill_between(xs, ivrl_mean2 - ivrl_se2, ivrl_mean2 + ivrl_se2, color='lime', alpha=0.5)

        plt.plot(xs,ivrl_mean3, '-o', color='skyblue', label = '92 nodes')
        plt.fill_between(xs, ivrl_mean3 - ivrl_se3, ivrl_mean3 + ivrl_se3, color='dodgerblue', alpha=0.5)

        plt.plot(xs,ivrl_mean4, '-o', color='olive', label = '128 nodes')
        plt.fill_between(xs, ivrl_mean4 - ivrl_se4, ivrl_mean4 + ivrl_se4, color='olive', alpha=0.5)
   


        plt.margins(x=0.1, y=0.2) 
        plt.xlabel("SOFA score" + ("" if num == 0 else " organ " + str(num)), fontsize=17)
        plt.ylabel("Mean Dosage", fontsize=17)
        plt.legend(fontsize=17)
        plt.title('IV fluids', fontsize=17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)

        plt.subplot(122)

        plt.plot(xs,vccl_mean, '-o', color='skyblue', label = 'Clinician policy')
        plt.fill_between(xs, vccl_mean - vccl_se, vccl_mean + vccl_se, color='skyblue',alpha=0.5)

        plt.plot(xs,vcrl_mean, '-o', color='royalblue', label = '16 nodes')
        plt.fill_between(xs, vcrl_mean - vcrl_se, vcrl_mean + vcrl_se, color='royalblue', alpha=0.5)

        plt.plot(xs,vcrl_mean2, '-o', color='aquamarine', label = '42 nodes')
        plt.fill_between(xs, vcrl_mean2 - vcrl_se2, vcrl_mean2 + vcrl_se2, color='aquamarine', alpha=0.5)

        plt.plot(xs,vcrl_mean3, '-o', color='teal', label = '92 nodes')
        plt.fill_between(xs, vcrl_mean3 - vcrl_se3, vcrl_mean3 + vcrl_se3, color='teal', alpha=0.5)

        plt.plot(xs,vcrl_mean4, '-o', color='slategray', label = '128 nodes')
        plt.fill_between(xs, vcrl_mean4 - vcrl_se4, vcrl_mean4 + vcrl_se4, color='slategray', alpha=0.5)


        plt.margins(x=0.1, y=0.2)  
        plt.xlabel("SOFA score" + ("" if num == 0 else " organ " + str(num)), fontsize=17)
        plt.ylabel("Mean Dosage", fontsize=17)
        plt.legend(fontsize=17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.title('Vasopressors', fontsize=17)
        plt.show()
        num += 1



   
    




    


    



    


    
