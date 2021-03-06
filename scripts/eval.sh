BACKBONE=resnet50

for TGT in {Jul,Augu,Sep,Oct,Nov,Dec,Jan2021,Feb2021,Mar2021,batch_2,batch_3}
do
    for best in {1,2,3,}
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
            --infer_model ./Domain_Generalization/save_models/tgt_Jan2021_src_Nex_trainingset_RSC_True_best${best}_AUC.pth
    done

    for last in {1,2,3,}
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
             --infer_model ./Domain_Generalization/save_models/tgt_Jan2021_src_Nex_trainingset_RSC_True_last${last}_AUC.pth
     done
 done
