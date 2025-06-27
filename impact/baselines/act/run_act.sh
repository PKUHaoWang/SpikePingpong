python3 train_act.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir ./checkpoints/pos_1 \
--policy_class ACT --kl_weight 10 --chunk_size 1 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0 