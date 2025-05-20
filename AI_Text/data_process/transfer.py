import json

def read_data(path):
    # 按行读取 JSON 文件
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

instruction = '''你是一位擅长文本辨别的专家，你的任务是分析一段给定的中文文本，并判断其是AI生成还是人类编写。如果是AI生成的文本，请返回“AI文本”；如果是人类编写的文本，请返回“人工文本”。
现在请针对以下输入的文本给出输出：\n'''

def lable_to_text(label):
    if label == 1:
        return 'AI文本'
    elif label == 0:
        return '人工文本'
    else:
        raise ValueError("Invalid label")

def rejected_label(label):
    if label == 1:
        return '人工文本'
    elif label == 0:
        return 'AI文本'
    else:
        raise ValueError("Invalid label")

def transfer_train_dpo(data, save_path=None):
    # reward_data = []
    # ppo_data = []
    dpo_data = []
    for entry in data:
        text = entry['text']
        label = lable_to_text(entry['label'])
        reject_label = rejected_label(entry['label'])
        context = instruction + text
        dpo_data.append({
            'conversations': [{"from": "human", "value": context}],
            'chosen': {"from": "gpt", "value": label},
            'rejected': {"from": "gpt", "value": reject_label},
        })
        # ppo_data.append({
        #     'instruction': context,
        #     'input': "",
        #     'output': target
        # })
    # 将数据保存为 JSON 文件
    # with open('./reward_train.json', 'w', encoding='utf-8') as f:
    #     json.dump(reward_data, f, ensure_ascii=False, indent=4)
    # with open('./ppo_train.json', 'w', encoding='utf-8') as f:
    #     json.dump(ppo_data, f, ensure_ascii=False, indent=4)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=4)

def transfer_train_sft(data, save_path=None):
    train_data = []
    for entry in data:
        text = entry['text']
        label = lable_to_text(entry['label'])
        context = instruction + text
        train_data.append({
            'instruction': context,
            'input': "",
            'output': label
        })
    # 将数据保存为 JSON 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

def transfer_test(data, save_path=None):
    test_data = []
    for entry in data:
        text = entry['text']
        label = lable_to_text(entry['label'])
        context = instruction + text
        test_data.append({
            'instruction': context,
            'label': label
        })
    # 将数据保存为 JSON 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


def transfer_last_test(data, save_path=None):
    last_test_data = []
    for entry in data:
        text = entry['text']
        id = entry['id']
        context = instruction + text
        last_test_data.append({
            'instruction': context,
            'text': text,
            'id': id
        })
    # 将数据保存为 JSON 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(last_test_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # transfer_train(read_data('../data/train.json'), save_path='../data/inst_data/dpo_train.json')
    # transfer_test(read_data('../data/dev.json'), save_path='../data/inst_data/inst_dev.json')
    # transfer_train_sft(read_data('../data/train.json'), save_path='../data/inst_data/sft_train.json')

    transfer_train_sft(read_data('./data/augmented_data/augmented_train.json'), save_path='./data/augmented_inst_data/sft_train.json')
    transfer_train_dpo(read_data('./data/augmented_data/augmented_train.json'), save_path='./data/augmented_inst_data/dpo_train.json')
    transfer_test(read_data('./data/dev.json'), save_path='./data/augmented_inst_data/inst_dev.json')
    transfer_last_test(read_data('./data/test.json'), save_path='./data/augmented_inst_data/inst_test.json')
    print("指令数据构造完成！已保存至 ./data/augmented_inst_data")