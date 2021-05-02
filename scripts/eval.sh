BACKBONE=resnet50

for best in {1,2,3,}
do
    for TGT in {batch_3,}
    do
        python -u "/home/hcaoaf/github/RSC/Domain_Generalization/train.py" \
            --eval_mode \
            -c 9 \
            -e 50 \
            -b 128 \
            --downsample_target \
            --cuda_number 2 \
            --target ${TGT}\
            --network ${BACKBONE} \
            --infer_model ./Domain_Generalization/save_models/tgt_Mar2021_src_Nex_trainingset_RSC_True_best${best}_FPR.pth
    done
done

# for last in {1,2,3,}
# do
#     for TGT in {Nex_trainingset,}
#     do
#         python -u "/home/hcaoaf/github/RSC/Domain_Generalization/train.py" \
#             --eval_mode \
#             -c 9 \
#             -e 50 \
#             -b 128 \
#             --downsample_target \
#             --cuda_number 2 \
#             --target ${TGT}\
#             --network ${BACKBONE} \
#             --infer_model ./Domain_Generalization/save_models/tgt_Mar2021_src_Nex_trainingset_RSC_True_last${last}_FPR.pth
#     done
# done
