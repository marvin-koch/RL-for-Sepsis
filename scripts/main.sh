model=$1

mode=$2

device=$3

if [ "$#" -ne 3 ]; then
    echo "Error: not enough arguments"
    exit 1
fi

if [ "$model" = "lstm" ]; then

    if [ "$mode" = "normal" ]; then
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16.pth --device $device          
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16.pth --device $device         
        python3 run_model.py --model lstm --task train --path lstm_td3_run_42.pth --device $device         
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_42.pth --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_92.pth --device $device         
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_92.pth --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_128.pth --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_128.pth --device $device
    elif [ "$mode" = "bias_loss" ]; then
        python3 run_model.py --model lstm --task train --path lstm_td3_run_bias_loss_5_1_16.pth --loss_param 5,1 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_bias_loss_5_1_16.pth --loss_param 5,1 --device $device  
        python3 run_model.py --model lstm --task train --path lstm_td3_run_bias_loss_1_5_16.pth --loss_param 1,5 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_bias_loss_1_5_16.pth --loss_param 1,5 --device $device  
    elif [ "$mode" = "reward_simple" ]; then
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_simple_1_16.pth --reward simple,1 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_simple_1_16.pth --reward simple,1 --device $device 
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_simple_2_16.pth --reward simple,2 --device $device       
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_simple_2.pth --reward simple,2 --device $device  
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_simple_3_16.pth --reward simple,3 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_simple_3_16.pth --reward simple,3 --device $device 
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_simple_4_16.pth --reward simple,4 --device $device       
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_simple_4_16.pth --reward simple,4 --device $device 
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_simple_5_16.pth --reward simple,5 --device $device       
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_simple_5_16.pth --reward simple,5 --device $device 
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_simple_6_16.pth --reward simple,6 --device $device
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_simple_6_16.pth --reward simple,6 --device $device 
    elif [ "$mode" = "reward_both" ]; then
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_both_1_16.pth --reward both,1 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_both_1_16.pth --reward both,1 --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_both_2_16.pth --reward both,2 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_both_2_16.pth --reward both,2 --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_both_3_16.pth --reward both,3 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_both_3_16.pth --reward both,3 --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_both_4_16.pth --reward both,4 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_both_4_16.pth --reward both,4 --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_both_5_16.pth --reward both,5 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_both_5_16.pth --reward both,5 --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_16_bias_both_6_16.pth --reward both,6 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_16_bias_both_6_16.pth --reward both,6 --device $device
    elif [ "$mode" = "reward_param" ]; then
        python3 run_model.py --model lstm --task train --path lstm_td3_run_reward_param_25_075_16.pth --reward_param 0.25,0.075 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_reward_param_25_075_16.pth --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_reward_param_15_175_16.pth --reward_param 0.15,0.175 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_reward_param_15_175_16.pth --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_reward_param_05_275_16.pth --reward_param 0.05,0.275 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_reward_param_05_275_16.pth --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_reward_param_3_025_16.pth --reward_param 0.3,0.025 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_reward_param_3_025_16.pth --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_reward_param_1_225_16.pth --reward_param 0.1,0.225 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_reward_param_1_225_16.pth --device $device
        python3 run_model.py --model lstm --task train --path lstm_td3_run_reward_param_2_125_16.pth --reward_param 0.2,0.125 --device $device        
        python3 run_model.py --model lstm --task eval --path lstm_td3_run_reward_param_2_125_16.pth --device $device
    else
        echo "Invalid mode: $mode"
    fi

elif [ "$model" = "transformer" ]; then
    if [ "$mode" = "normal" ]; then
        python3 run_model.py --model transformer --task train --path transformer_td3_run_1_1.pth --device $device  
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_1_1.pth --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_2_2.pth  --device $device
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_2_2.pth --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_4_4.pth --device $device
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_4_4.pth --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_2_4.pth --device $device 
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_2_4.pth --device $device
    elif [ "$mode" = "bias_loss" ]; then
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_loss_5_1_1_1.pth --loss_param 5,1 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_loss_5_1_1_1.pth --loss_param 5,1 --device $device  
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_loss_1_5_1_1.pth --loss_param 1,5 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_loss_1_5_1_1.pth --loss_param 1,5 --device $device  
    elif [ "$mode" = "reward_simple" ]; then
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_simple_1_1_1.pth --reward simple,1 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_simple_1_1_1.pth --reward simple,1 --device $device 
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_simple_2_1_1.pth --reward simple,2 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_simple_2_1_1.pth --reward simple,2 --device $device 
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_simple_3_1_1.pth --reward simple,3 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_simple_3_1_1.pth --reward simple,3 --device $device 
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_simple_4_1_1.pth --reward simple,4 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_simple_4_1_1.pth --reward simple,4 --device $device 
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_simple_5_1_1.pth --reward simple,5 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_simple_5_1_1.pth --reward simple,5 --device $device 
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_simple_6_1_1.pth --reward simple,6 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_simple_6_1_1.pth --reward simple,6 --device $device 
    elif [ "$mode" = "reward_both" ]; then
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_both_1_1_1.pth --reward both,1 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_both_1_1_1.pth --reward both,1 --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_both_2_1_1.pth --reward both,2 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_both_2_1_1.pth --reward both,2 --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_both_3_1_1.pth --reward both,3 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_both_3_1_1.pth --reward both,3 --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_both_4_1_1.pth --reward both,4 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_both_4_1_1.pth --reward both,4 --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_both_5_1_1.pth --reward both,5 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_both_5_1_1.pth --reward both,5 --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_bias_both_6_1_1.pth --reward both,6 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_bias_both_6_1_1.pth --reward both,6 --device $device
    elif [ "$mode" = "reward_param" ]; then
        python3 run_model.py --model transformer --task train --path transformer_td3_run_reward_param_25_075_1_1.pth --reward_param 0.25,0.075 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_reward_param_25_075_1_1.pth --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_reward_param_15_175_1_1.pth --reward_param 0.15,0.175 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_reward_param_15_175_1_1.pth --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_reward_param_05_275_1_1.pth --reward_param 0.05,0.275 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_reward_param_05_275_1_1.pth --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_reward_param_3_025_1_1.pth --reward_param 0.3,0.025 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_reward_param_3_025_1_1.pth --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_reward_param_1_225_1_1.pth --reward_param 0.1,0.225 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_reward_param_1_225_1_1.pth --device $device
        python3 run_model.py --model transformer --task train --path transformer_td3_run_reward_param_2_125_1_1.pth --reward_param 0.2,0.125 --device $device        
        python3 run_model.py --model transformer --task eval --path transformer_td3_run_reward_param_2_125_1_1.pth --device $device
    else
        echo "Invalid mode: $mode"
    fi
else
    echo "Invalid model: $model"
fi