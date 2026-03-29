# Self
CUDA_VISIBLE_DEVICES=4,5,6,7 python moco_gsv.py \
  -a vit_base -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10002' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
    self_cities10_1m.pkl


# Spatial
CUDA_VISIBLE_DEVICES=0,1,2,3 python moco_gsv.py \
  -a vit_base -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10005' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  spatial_5k100t_cities10_1m_new.pkl 

# Temporal
CUDA_VISIBLE_DEVICES=4,5,6,7  python moco_gsv.py \
  -a vit_base -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10003' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  temporal_random_cities10t_1m.pkl