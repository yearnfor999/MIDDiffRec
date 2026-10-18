cd ..
#python main.py --data_name Beauty --model_idx 1  --diff_weight 1 --cl_weight 0.2 --timesteps 50 --intent_num 32 --w 2
python main.py --data_name Beauty --model_idx 1  --diff_weight 1 --user_weight 0.05 --diffusion_rec_weight 0.05 --timesteps 50 --intent_num 16 --w 2 --use_multi_intent --nmc 3 --mc_func mean