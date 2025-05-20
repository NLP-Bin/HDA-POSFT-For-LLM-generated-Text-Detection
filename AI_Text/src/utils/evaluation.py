import torch
from tqdm import tqdm
import re
import json
import os
import time
from sklearn.metrics import classification_report

from utils.logger import get_logger
from utils.predict import Model


logger = get_logger(__name__)

def parser_outputs(s):
    """
    Parse the outputs
    """
    if "AI" in s:
        return "AI文本"
    elif "人工" in s:
        return "人工文本"

def evaluate_dev_data(test_data, model_name_or_path, checkpoint_dir, template, temperature=0.1, top_p=1.0, finetuning_type="lora"):
    """
    Compute scores given the predictions and gold labels
    """
    model = Model(model_name_or_path, checkpoint_dir, template, temperature, top_p, finetuning_type)

    trues, preds = [], []
    all_results = []  # 用于保存全部记录
    error_records = []  # 用于保存错误的记录
    error_id = 0  # 记录错误记录的条数
    for data in tqdm(test_data, desc="Predicting the Test_Data"):
        instruction = data['instruction']

        message = [
            {"role": "user", "content": instruction}
        ]
        pred = model.generate(message)[0].response_text
        true = data['label']
        # print("===================================")
        # print(f"Instruction: {instruction}")
        # print(f"True: {true}")
        # print(f"Original Pred: {pred}")
        # 解析输出
        pred = parser_outputs(pred)
        true = parser_outputs(true)


        # print(f"Processed Pred: {pred}")

        preds.append(pred)
        trues.append(true)

        all_results.append({
            # 'sent_id': data['id'],
            'instruction': instruction,
            'true': true,
            'pred': pred
        })
        # 如果预测的输出与真实的输出不匹配，将它们添加到错误记录中
        if pred != true:
            error_id += 1
            error_records.append({
                'error_id': str(error_id),
                # 'sent_id': data['id'],
                'instruction': instruction,
                'true': true,
                'pred': pred
            })
            print("===================================")
            print(f"Instruction: {instruction}")
            print(f"True: {true}")
            print(f"Pred: {pred}")

    if not os.path.exists(f'./results/'):
        os.makedirs(f'./results/')

    results_path = f"./results/all_results_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    # 保存错误记录到json文件
    if error_records:
        error_path = f"./results/error_records_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_records, f, ensure_ascii=False, indent=4)


    scores = compute_scores(preds, trues)
    return scores


def evaluate_test_data(test_data, model_name_or_path, checkpoint_dir, template, temperature=0.1, top_p=1.0, finetuning_type="lora"):
    """
    Compute scores given the predictions and gold labels
    """
    model = Model(model_name_or_path, checkpoint_dir, template, temperature, top_p, finetuning_type)
    all_results = []  # 用于保存全部记录
    label_mapping = {"人工文本": 0, "AI文本": 1}  # 预定义标签映射
    for data in tqdm(test_data, desc="Predicting the Test_Data"):
        instruction = data['instruction']
        message = [
            {"role": "user", "content": instruction}
        ]
        pred = model.generate(message)[0].response_text

        print("===================================")
        print(f"Instruction: {instruction}")
        print(f"Original Pred: {pred}")
        # 解析输出
        pred = parser_outputs(pred)
        print(f"Parsing Pred: {pred}")
        #输出映射回0，1。如果不在映射中，则认为是异常，设置为 1（AI文本）
        pred_label = label_mapping.get(pred, 1)  # 预测失败或异常情况设为 1
        print(f"Mapping Pred: {pred_label}")
        all_results.append({
            # 'sent_id': data['id'],
            'id': data["id"],
            'text': data["text"],
            'lable': pred_label
        })


    if not os.path.exists(f'./results/'):
        os.makedirs(f'./results/')

    results_path = f"./results/all_results_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)


def compute_scores(preds, trues):
    """
    Compute scores given the predictions and gold labels
    """
    report = classification_report(trues, preds, target_names=['人工文本', 'AI文本'], output_dict=True, digits=4)

    # 打印分类报告
    print(report)

    # 获取第一个类别（人工文本）的指标
    artificial_text_metrics = report['人工文本']
    precision_artificial = artificial_text_metrics['precision']
    recall_artificial = artificial_text_metrics['recall']
    f1_artificial = artificial_text_metrics['f1-score']

    # 获取第二个类别（AI文本）的指标
    ai_text_metrics = report['AI文本']
    precision_ai = ai_text_metrics['precision']
    recall_ai = ai_text_metrics['recall']
    f1_ai = ai_text_metrics['f1-score']

    # print(f"人工文本 - P: {precision_artificial}, R: {recall_artificial}, F1: {f1_artificial}")
    # print(f"AI文本 - P: {precision_ai}, R: {recall_ai}, F1: {f1_ai}")
    # 返回‘人工文本’的精确率、召回率和F1分数
    return {
        'precision_人工文本': precision_artificial,
        'recall_人工文本': recall_artificial,
        'f1_人工文本': f1_artificial,
        'precision_AI文本': precision_ai,
        'recall_AI文本': recall_ai,
        'f1_AI文本': f1_ai,
        'accuracy': report['accuracy'],
        'macro avg': report['macro avg'],
        'weighted avg': report['weighted avg']
    }



