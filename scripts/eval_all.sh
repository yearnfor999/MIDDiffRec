cd ..
python main.py --data_name Beauty --model_idx 1  --diff_weight 1 --cl_weight 0.2 --timesteps 50 --intent_num 32 --w 2 --eval_only
python main.py --data_name Toys_and_Games  --model_idx 1 --diff_weight 0.2 --cl_weight 0.2 --timesteps 200   --intent_num 1024 --w 2 --eval_only
python main.py --data_name Sports_and_Outdoors --model_idx 1 --diff_weight 1 --cl_weight 0.4 --timesteps 200   --intent_num 256 --w 2 --eval_only
python main.py --data_name Video --model_idx 1  --diff_weight 1 --cl_weight 0.8 --timesteps 100 --intent_num 128 --w 2 --hidden_dropout_prob 0.4 --attention_probs_dropout_prob 0.4 --eval_only 
python main.py --data_name ml-1m --model_idx 1 --hidden_dropout_prob 0.1 --lr 0.001  --diff_weight 1 --cl_weight 0.2 --attention_probs_dropout_prob 0.1 --timesteps 50 --intent_num 1024 --eval_only
