

python /home/liyong/code/svpretrain/moco-v3/main_test_rs_feature.py \
    --data_path /data_nas/liyong/GoogleEarth/metadata/US/self_la_latest_GEOID.pkl \
    --output_dir /home/liyong/code/svpretrain/output_dir/feature/google_earth/Mocov3VITB-spatial_cbg_la_postpretrain_80w.pkl \
    --model_path /home/liyong/code/svpretrain/output_dir/checkpoint/GoogleEarth/Mocov3VITB-spatial_cbg_la_postpretrain_80w/checkpoint_99.pth.tar \
    --gpu_id 4 

python /home/liyong/code/svpretrain/moco-v3/main_test_bsv1m_feature.py \
--data_path /data_nas/lsr/BaiduSvs_history/output/metadata/success/Fujian/Fuzhou_exist.csv  \
--model_path /home/liyong/code/svpretrain/output_dir/checkpoint/mocov3_paper/vit-b-300ep.pth.tar \
--output_dir  /data_nas/dailing/alphapop/data/streetview/fuhzou_embedding.csv \
--batch_size 256 --gpu_id 0 