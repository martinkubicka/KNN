# KNN Project

### Setup
```
1. ssh xlogin00@perian.grid.cesnet.cz
2. Copy/clone whole project to metacentrum (for example: scp -r * xlogin00@perian.grid.cesnet.cz:~/KNN/)
3. Download dataset or copy it with ssh (in the root folder of the project - where Dockerfile is located)
4. Allocate GPU (for example: qsub -I -l select=1:ncpus=1:mem=32gb:ngpus=1:gpu_mem=16gb:scratch_local=32gb -l walltime=100:0:0 -q gpu_long)
5. cd path/to/project (for example: /storage/brno2/home/xlogin00/KNN)
6. chmod +x setup.sh test_baseline.sh train_baseline.sh
7. ./setup.sh
```

### Training (baseline)
```
1. You can change setting like batch size, paths etc. in globals.py
2. ./train_baseline
```

### Testing (baseline)
```
1. You can change setting like batch size, paths etc. in globals.py
2. ./test_baseline
```

### Disown terminal 
Disown interactive terminal so you can close it. Output will be redirected to `output.txt`.
```
nohup python -u ./src/architectures/baseline/train.py > output.txt 2>&1 &
disown
```

### TODOs
- `wandb` support
