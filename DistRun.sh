#!/bin/bash
source /home/fanxiangyu/env3.10/bin/activate

#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
# 停掉之前的占用进程
echo "Killing mess/testNorm.py..."
pkill -f testNorm.py

echo "Waiting for GPU to be released..."
sleep 5


# 获取当前时间（格式：YYYY-MM-DD_HH-MM-SS）
CURRENT_TIME=$(date +"%Y-%m-%d")
# 定义日志目录和文件名
LOG_DIR="logByFxy/DistLog"
LOG_FILE="${LOG_DIR}/big_tensor_${CURRENT_TIME}.log"
# 执行 Python 命令，并将 stdout 和 stderr 重定向到日志文件

# --api_config_file="big_tensor_0718_normalize.txt" \

# 向日志文件写入一行50个等号作为分隔符
echo -e "\n\n\n" >> "${LOG_FILE}" 2>&1

echo "=================测试开始======================测试开始====================测试开始=========================测试开始====================测试开始========================测试开始=================================" >> "${LOG_FILE}" 2>&1
echo "begin test!"



echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES}" >> "${LOG_FILE}" 2>&1
# python engineV2.py --api_config='paddle.dist(x=Tensor([4294967297],"float16"), y=Tensor([4294967297],"float16"), )' --accuracy=True 
#是否将测试结果保存为npz文件，1为保存，0为不保存
export SAVE_PNORM_DATA=0
echo "SAVE_PNORM_DATA:${SAVE_PNORM_DATA}" >> "${LOG_FILE}" 2>&1


# python engine.py \
#     --api_config_file='logByFxy/api_config_oom.txt' \
#     --accuracy=True \
#     >> "${LOG_FILE}" 2>&1

# python engineV2.py \
#     --api_config_file='big_tensor_0718_normalize.txt' \
#     --accuracy=True \
#     --log_dir="$LOG_DIR"\
#     >> "${LOG_FILE}" 2>&1

# 2147483647
    # --api_config='paddle.digamma(Tensor([10, 42949673, 10],"float16"), )' \




python engine.py \
    --api_config='paddle.dist(x=Tensor([10],"float32"), y=Tensor([429496730, 10],"float32"), p=4, )' \
    --accuracy=True \
    >> "${LOG_FILE}" 2>&1
# python engine.py \
#     --api_config_file='big_tensor_0718_normalize.txt' \
#     --accuracy=True \
#     >> "${LOG_FILE}" 2>&1


# echo "开始测试异常数据"
# python normTestnpz.py >> "${LOG_FILE}" 2>&1

echo "Log file saved to: ${LOG_FILE}"
# 打印日志文件路径，方便查看
echo "Restarting normTest occupancy..."
nohup python mess/testNorm.py --gpu ${CUDA_VISIBLE_DEVICES} --mem 60 > occupy.log 2>&1 &

echo "test end!"

# PYTHON_PID=$!

# sleep 1
# if ! ps -p "$PYTHON_PID" > /dev/null; then
#     echo "错误：engine 启动失败，请检查 $LOG_FILE"
#     exit 1
# fi

# echo -e "\n\033[32m执行中... 另开终端运行监控:\033[0m"
# echo -e "1. GPU使用:   watch -n 1 nvidia-smi"
# echo -e "2. 日志目录:  ls -lh $LOG_DIR"
# echo -e "3. 详细日志:  tail -f $LOG_FILE"
# echo -e "4. 终止任务:  kill $PYTHON_PID"
# echo -e "\n进程已在后台运行，关闭终端不会影响进程执行"

exit 0