#/bin/bash
docker run -it --rm -d \
--gpus all \
--shm-size=64g \
-p 9999:9999 \
-v $(pwd)/data:/workspace/data \
-v $(pwd)/result:/workspace/result \
-v $(pwd)/save_model:/workspace/save_model \
-v $(pwd)/src:/workspace/src \
-v $(pwd)/environment.yml:/workspace/environment.yml \
-v $(pwd)/requirments.txt:/workspace/requirments.txt \
-v /home/v-jianghao/.cache:/root/.cache \
resrchvc4cr.azurecr.io/v-jianghao/research:llm

echo $(pwd)/src

# --gpus '"device=0"' \