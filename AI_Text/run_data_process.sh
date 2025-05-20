#对人工文本进行数据增强
python ./data_process/augment.py
#将数据集转换为训练格式，SFT格式, DPO格式
python ./data_process/transfer.py