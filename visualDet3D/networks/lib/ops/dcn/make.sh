export CUDA_HOME=/usr/local/cuda        #"/home/rispro-sils/ADAS_Kedaireka/env/cuda-11.8/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda #"/home/rispro-sils/ADAS_Kedaireka/env/cuda-11.8/"
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
python3 setup.py build_ext --inplace