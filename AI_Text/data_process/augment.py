import json
import random
import re
from copy import deepcopy

# 语气增强词汇
subjective_phrases = ['说实话', '讲真', '我觉得', '说真的', '老实说']
modal_particles = ['嘛', '吧', '哈', '呀', '啦']

# 同音/打字错误映射
typo_map = {
    '的': '地', '是': '事', '在': '再', '有': '又',
    '不': '布', '了': '啦', '很': '狠', '好': '号',
    '看': '砍', '吃': '痴'
}

# 高频词汇（用于重复）
high_freq_adj = ['很好', '非常', '特别', '真的', '不错']

def insert_subjective(text):
    """插入主观修饰词和语气助词"""
    sentences = re.split(r'(。|！|\!|？|\?)', text)
    result = ""
    for i in range(0, len(sentences) - 1, 2):
        sent = sentences[i]
        end = sentences[i+1]
        if random.random() < 0.5:
            sent = random.choice(subjective_phrases) + '，' + sent
        if random.random() < 0.5:
            sent += random.choice(modal_particles)
        result += sent + end
    return result

def inject_errors(text):
    """错误注入"""
    chars = list(text)
    new_chars = []
    i = 0
    while i < len(chars):
        c = chars[i]
        # 同音错字
        if c in typo_map and random.random() < 0.1:
            new_chars.append(typo_map[c])
        # 多余标点
        elif c in '，。！？' and random.random() < 0.1:
            new_chars.append(c + random.choice(['！', '……', '。']))
        # 正常字符
        else:
            new_chars.append(c)
        # 重复词
        if i < len(chars)-1 and chars[i:i+2][0]+chars[i:i+2][1] in high_freq_adj and random.random() < 0.3:
            new_chars.extend(chars[i:i+2])
            i += 2
            continue
        i += 1
    # 语病模板（如：我感到我很开心 -> 我我感到很开心）
    if random.random() < 0.2:
        words = text.split(' ')
        if len(words) >= 2:
            insert_idx = random.randint(0, len(words)-1)
            words.insert(insert_idx, words[insert_idx])
            return ''.join(new_chars) + ' ' + ' '.join(words)
    return ''.join(new_chars)

def shuffle_sentences(text, shuffle_ratio=0.5):
    """顺序扰动"""
    sentences = re.split(r'(。|！|\!|？|\?)', text)
    combined = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
    if len(combined) <= 1:
        return text
    shuffle_count = int(len(combined) * shuffle_ratio)
    indices = list(range(len(combined)))
    selected = random.sample(indices, shuffle_count)
    shuffled = combined[:]
    to_shuffle = [combined[i] for i in selected]
    random.shuffle(to_shuffle)
    for idx, new_val in zip(selected, to_shuffle):
        shuffled[idx] = new_val
    return ''.join(shuffled)

def augment_text(text, method):
    if method == 'tone':
        return insert_subjective(text)
    elif method == 'error':
        return inject_errors(text)
    elif method == 'shuffle':
        return shuffle_sentences(text)
    else:
        return text

def augment_dataset(data):
    # 过滤出 human 文本
    human_texts = [item for item in data if item['label'] == 0]
    other_texts = [item for item in data if item['label'] != 0]

    # 打乱后分组
    # random.shuffle(human_texts)
    total = len(human_texts)
    n_tone = int(total * 0.3)
    n_error = int(total * 0.4)
    n_shuffle = total - n_tone - n_error

    augmented_items = []

    for i, item in enumerate(human_texts):
        aug_item = deepcopy(item)
        if i < n_tone:
            aug_item['text'] = augment_text(item['text'], 'tone')
            aug_item['aug_type'] = 'tone'
        elif i < n_tone + n_error:
            aug_item['text'] = augment_text(item['text'], 'error')
            aug_item['aug_type'] = 'error'
        else:
            aug_item['text'] = augment_text(item['text'], 'shuffle')
            aug_item['aug_type'] = 'shuffle'
        augmented_items.append(aug_item)

    # 原始数据 + 增强后的新数据
    return data + augmented_items

# 示例运行
if __name__ == '__main__':
    with open('./data/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    augmented = augment_dataset(data)

    with open('./data/augmented_data/augmented_train.json', 'w', encoding='utf-8') as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print("数据增强完成！已保存至 ./data/augmented_data/augmented_train.json")
