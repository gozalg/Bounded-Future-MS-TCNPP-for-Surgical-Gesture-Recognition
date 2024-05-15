python3 Trainer.py --data_path data/JIGSAWS/Suturing/frames/ \
                            --transcriptions_dir data/JIGSAWS/Suturing/transcriptions/ \
                            --epochs 2 \
                            --eval_freq 1 \
                            --save_freq 100 \
                            --workers 8 \
                            --arch 'EfficientnetV2' \
                            --gpu_id 0 \
                            --exp 'EfficientNetV2-m-jigsaws-additional-2048' \
                            --additional_param_num 2048

#python3 trainer.py --data_path all_datasets/jigsaws/data/Suturing/frames/ \
#                             --transcriptions_dir all_datasets/jigsaws/data/Suturing/transcriptions/ \
#                             --epochs 100 \
#                             --save_freq 100 \
#                             --workers 8 \
#                             --arch '2D-ResNet-18' \
#                             --gpu_id 1 \
#                             --preload False \
#                             --exp 'resnet18-jigsaws-additional-2048' \
#                             --additional_param_num 2048


# python3 trainer.py --data_path all_datasets/jigsaws/data/Suturing/frames/ \
#                              --transcriptions_dir all_datasets/jigsaws/data/Suturing/transcriptions/ \
#                              --epochs 130 \
#                              --save_freq 100 \
#                              --workers 8 \
#                              --arch '2D-ResNet-18' \
#                              --gpu_id 1 \
#                              --preload False \
#                              --exp 'resnet18-jigsaws-no-augmentation-sample' 
#                              --do_horizontal_flip True \
#                              --do_vertical_flip True \
#                              --do_color_jitter True
                            