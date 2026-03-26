    python3 main.py --model UNCERTHED --fine_tuning --dataset_name UNCERT_BSDS --lr 1e-6 --max_epoch 5 --dataset_folder /home/mas11/Documents/Datasets/BSDS\
        --output without_octave
    python3 main.py --model UNCERTHEDOCT --fine_tuning --dataset_name UNCERT_BSDS --lr 1e-6 --max_epoch 5 --dataset_folder /home/mas11/Documents/Datasets/BSDS\
            --alpha 0.5 --output testing