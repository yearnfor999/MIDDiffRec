import copy
import random
import torch

from torch.utils.data import Dataset
from utils import generate_rating_matrix_valid, generate_rating_matrix_test


def D(i_file,o_file,max_len):

    with open(i_file,"r+") as fr:
        data=fr.readlines()
    aug_d={}
    # training, validation, and testing
    max_save_len=max_len+3
    # save
    max_keep_len=max_len+2
    for d_ in data:
        u_i,item=d_.split(' ',1)
        item=item.split(' ')
        item[-1]=str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start=0
        j=3
        if len(item)>max_save_len:
            # training, validation, and testing
            while start<len(item)-max_keep_len:
                j=start+4
                while j<len(item):
                    if start<1 and j-start<max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j+=1
                    else:
                        aug_d[u_i].append(item[start:start+max_save_len])
                        break
                start+=1
        else:
            while j<len(item):
                aug_d[u_i].append(item[start:j+1])
                j+=1
    with open(o_file,"w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i+" "+' '.join(i_)+"\n")

def D_random(i_file, o_file, max_len):

    with open(i_file, "r+") as fr:
        data = fr.readlines()
    
    aug_d = {}
    # training, validation, and testing
    max_save_len = max_len + 3

    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        item[-1] = str(eval(item[-1]))  # Ensure the last item is a valid number
        aug_d.setdefault(u_i, [])
        
        start = 0
        j = 3
        if len(item) > max_save_len:
            # training, validation, and testing
            while start < len(item) - (max_len + 2):
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < max_save_len:
                        random_sample = random.sample(item[start:j], len(item[start:j]))
                        aug_d[u_i].append(random_sample)
                        j += 1
                    else:
                        random_sample = random.sample(item[start:start + max_save_len], len(item[start:start + max_save_len]))
                        aug_d[u_i].append(random_sample)
                        break
                start += 1
        else:
            while j < len(item):
                random_sample = random.sample(item[start:j + 1], len(item[start:j + 1]))
                aug_d[u_i].append(random_sample)
                j += 1

    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")

def get_seqs_and_matrixes(type, data_file):
    user_seq = [] # 收集每行的交互序列
    user_id = [] # 收集每行的用户ID
    item_set = set() # set()去重，集合会自动去掉重复的元素    建立一个全局物品 ID 集合，方便统计最大物品编号
    
    with open(data_file, "r") as fr:
        for line in fr:
            parts = line.strip().split() # .strip() 会去掉字符串首尾的空白字符（包括 空格、\t、\n、\r 等）  .split() 会按照任意空白字符（空格、制表符、换行符等）切分成一个列表
            user = int(parts[0]) # 行首是用户ID
            items = list(map(int, parts[1:])) # map 会把 int() 这个函数作用到 parts[1:] 里的每个元素上，把字符串型的物品ID转成整数
            user_id.append(user)
            user_seq.append(items)
            item_set.update(items)
    
    max_item = max(item_set) # 从全体物品集合里取最大物品ID  12101
    num_users = len(user_id) # 用户数 Beauty_s.txt：131413  Beauty.txt：22363
    num_items = max_item + 2 # 物品总“词表”大小 = 最大ID + 2   12103
    
    if type == "training":
        return user_id, user_seq
    elif type == "rating":
        valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
        test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
        return user_id, user_seq, max_item, valid_rating_matrix, test_rating_matrix
    else:
        raise NotImplementedError


class DatasetForInDiRec(Dataset):

    def __init__(self, args, user_seq, user_ids=None, test_neg_items=None, data_type="train"):
        self.args = args
        # user_seq：序列集合
        # 训练时传的是切好的前缀片段，来自 *_s.txt
        # 验证/测试时传的是原始完整序列（来自原始数据文件）
        self.user_seq = user_seq
        
        # 新增：用户ID列表（用于长期兴趣建模）
        self.user_ids = user_ids if user_ids is not None else list(range(len(user_seq)))
        
        self.test_neg_items = test_neg_items
        # data_type："train" | "valid" | "test" 决定取样切片策略
        self.data_type = data_type
        self.max_len = args.max_seq_length


    # __getitem__ 被 DataLoader 调用
    # DataLoader 会自动做这些事：
    # 先根据 __len__ 知道一共有多少样本；
    # 再按 batch_size 生成一批索引（比如 [0,1,2,3]）；
    # 然后逐个调用 dataset.__getitem__(0), dataset.__getitem__(1), dataset.__getitem__(2), dataset.__getitem__(3)；
    # 把返回的样本打包（collate），组成一个 batch
    def __getitem__(self, index):
        # 获取真实的用户ID（用于长期兴趣建模）
        real_user_id = self.user_ids[index]  # 真实用户ID
        sample_index = index  # 样本索引（保持原有逻辑）
        item_seq = self.user_seq[sample_index]

        # args.max_seq_length = 6（方便演示 padding）
        # 当前样本（下标 index）对应的序列
        # item_seq = [7, 11, 23, 35, 46, 58, 72]（长度 T=7）
        if self.data_type == "train":
            # 去掉末尾 3 个
            # [7, 11, 23, 35]
            input_seq = item_seq[ : -3]
            # [11, 23, 35, 46]
            # target_pos 与 input_seq 等长，是它整体右移一位：7→11，11→23，23→35，35→46
            target_pos = item_seq[1 : -2]
            # answer 取 倒数第三（46），恰好等于 target_pos 的最后一个（target_pos[-1]）
            answer = item_seq[-3]

        elif self.data_type == "valid":
            # 去掉末尾 2 个
            # [7, 11, 23, 35, 46]
            input_seq = item_seq[:-2]
            # [11, 23, 35, 46, 58]
            target_pos = item_seq[1:-1]
            # 取倒数第二 (58)
            answer = [item_seq[-2]]
        
        else: # 测试集
            #  Robustness to Noisy Data.
            # 调用“插噪声”函数，把原序列扩展为包含噪声的序列
            # 例子里返回：[7, 11, 99, 23, 35, 46, 101, 58, 72]
            item_seq_with_noise = self._add_noise_for_robustness(item_seq)
            # 去掉末尾 1 个
            # 取除最后一项以外的所有元素，作为模型的输入序列（teacher forcing 的标准做法）
            # input_seq = [7, 11, 99, 23, 35, 46, 101, 58]
            input_seq = item_seq_with_noise[:-1]
            # 取从第二项开始到末尾，作为“下一个物品”的逐位标签（shift 一位对齐）
            # 例子：target_pos = [11, 99, 23, 35, 46, 101, 58, 72]（长度 8）
            # 你可以看到：input_seq[i] 的目标就是 target_pos[i]
            target_pos = item_seq_with_noise[1:]
            # 最后一个
            # 取整条带噪序列的最后一个元素，单独作为 answer = (72)
            answer = [item_seq_with_noise[-1]]

        # 交给 _data_construction 去左侧 padding 到统一长度、再转成 tensor
        # 传递真实用户ID而不是样本索引
        cur_rec_tensors = self._data_construction(real_user_id, input_seq, target_pos, answer)

        return cur_rec_tensors


    # padding and to tensor
    # 举例：
    # user_id = 123（仅举例）
    # self.max_len = 10（方便演示“左侧补 0”）
    def _data_construction(self, user_id, input_seq, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        # 拷贝一份输入序列，防止原列表被原地修改
        # 此时：copied_input_seq = [7, 11, 99, 23, 35, 46, 101, 58]
        copied_input_seq = copy.deepcopy(input_seq)
        # 记录补零前的真实长度，后面做 mask/有效长度时要用
        # 例子：pad_len = 10 - 8 = 2
        input_seq_len = len(copied_input_seq)
        # 需要补多少个 0 到左侧，使长度对齐到 max_len
        # 例子：pad_len = 10 - 8 = 2
        pad_len = self.max_len - input_seq_len
        # 左侧补 0：[0, 0] + [7, 11, 99, 23, 35, 46, 101, 58]
        # 得到：copied_input_seq = [0, 0, 7, 11, 99, 23, 35, 46, 101, 58]（长度 10）
        copied_input_seq =[0] * pad_len+copied_input_seq
        # 之后再截取 最后 max_len 个元素，用于“既能补、也能截”的统一写法（当原序列超过 max_len 时会生效）
        copied_input_seq=copied_input_seq[-self.max_len:]

        # padding
        # 对 target_pos 采取相同的左侧补零和尾部截断策略
        # 原 target_pos = [11, 99, 23, 35, 46, 101, 58, 72]
        # 左补两个 0 → [0, 0, 11, 99, 23, 35, 46, 101, 58, 72]（长度 10）
        target_pos =  [0] * pad_len+target_pos
        target_pos = target_pos[-self.max_len:]

        assert len(target_pos) == self.max_len
        assert len(copied_input_seq) == self.max_len

        # to tensor
        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing  第 index 个样本
            torch.tensor(copied_input_seq, dtype=torch.long), # padding 后的序列
            torch.tensor(input_seq_len, dtype=torch.long), # 补零前的长度
            torch.tensor(target_pos, dtype=torch.long), # padding 后的 target_pos
            torch.tensor(answer, dtype=torch.long), # 目标
        )

        return cur_rec_tensors

    def _add_noise_for_robustness(self, items):
        # 如果噪声比例是 0，直接原样返回，不做任何修改
        if self.args.noise_ratio == 0:
            return items
        copied_sequence = copy.deepcopy(items) # 拷贝一份，避免原序列被原地修改
        # 计算要插入的噪声个数（向下取整）
        insert_nums = int(self.args.noise_ratio * len(copied_sequence))

        # 假设原序列 [7,11,23,35,46,58,72]
        # 假设参数：
        # noise_ratio = 0.3 → insert_nums = int(0.3*7) = 2（插 2 个）
        # item_size 取一个足够大（比如 item_size=1000，只要 ≥ max_item+2 即可）
        # insert_idx = random.choices(range(7), k=2)
        # 假设随机结果为：insert_idx = [2, 5]
        # 索引 2 对应的元素是 23
        # 索引 5 对应的元素是 58
        # 含义：“在索引 2 和 5 的元素之前各插入一个噪声 ID”。
        # 遍历并构造新序列
        # 初始：inserted_sequence = []
        # index=0, item=7
        # 0 不在 [2,5] → 不插噪
        # 追加当前物品：[7]
        # index=1, item=11
        # 1 不在 [2,5] → 不插噪
        # 追加当前物品：[7, 11]
        # index=2, item=23
        # 2 在 [2,5] → 需要先插噪
        # 随机一个不在原序列中的 ID，例如抽到 99（不在 [7,11,23,35,46,58,72]）
        # 插入噪声：[7, 11, 99]
        # 再追加当前物品：[7, 11, 99, 23]
        # index=3, item=35
        # 3 不在 [2,5] → 不插噪
        # 追加当前物品：[7, 11, 99, 23, 35]
        # index=4, item=46
        # 4 不在 [2,5] → 不插噪
        # 追加当前物品：[7, 11, 99, 23, 35, 46]

        # index=5, item=58
        # 5 在 [2,5] → 需要先插噪
        # 随机一个不在原序列中的 ID，例如抽到 101（也不在原序列）
        # 插入噪声：[7, 11, 99, 23, 35, 46, 101]
        # 再追加当前物品：[7, 11, 99, 23, 35, 46, 101, 58]

        # index=6, item=72
        # 6 不在 [2,5] → 不插噪
        # 追加当前物品：[7, 11, 99, 23, 35, 46, 101, 58, 72]
        # 最终返回：
        # inserted_sequence = [7, 11, 99, 23, 35, 46, 101, 58, 72]
        # 可以看到，原序列的相对顺序保持不变，只是在被选中的位置前面各插入了一个“没在原序列里出现过”的随机物品 ID（这里是 99 和 101）


        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = [] # 新序列容器
        for index, item in enumerate(copied_sequence):
            # 如果当前索引被选中，就先插入一个噪声物品 ID
            if index in insert_idx:
                # random.randint(1, self.args.item_size - 2): 避开 0（PAD）和 item_size-1（MASK），只抽真实物品 ID 范围
                # 0 是 PAD，不用；
                # item_size-1 是 MASK，也不用；
                # 因为 item_size = max_item + 2，所以这段范围正好覆盖所有真实物品 ID

                # 假设：
                # 数据集里最大物品 ID 是 58 (max_item = 58)
                # 词表大小 item_size = max_item + 2 = 60
                # | ID  | 含义
                # | --- | ----------------------
                # | 0   | PAD（序列补齐专用，不算物品）
                # | 1   | 物品 #1
                # | 2   | 物品 #2
                # | ... | ...
                # | 58  | 物品 #58（最大真实物品）
                # | 59  | MASK（掩码符号，用于训练时随机遮盖预测）
                item_id = random.randint(1, self.args.item_size - 2) # 噪声 Item
                # 如果随机到的 ID 本来就在原序列里，重新抽，保证“插入的噪声不是原序列里出现过的物品”
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            # 无论是否插入噪声，最后都要把当前真实物品也放进去
            inserted_sequence += [item]
        return inserted_sequence

    def __len__(self):

        return len(self.user_seq)


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader, SequentialSampler

    def get_args():
        parser = argparse.ArgumentParser()
        # system args
        parser.add_argument("--data_dir", default="./datasets/", type=str)
        parser.add_argument("--output_dir", default="output", type=str)
        parser.add_argument("--data_name", default="Beauty", type=str)


        parser.add_argument("--max_seq_length", type=int, default=50, 
                            help="max sequence length")

        parser.add_argument("--batch_size", type=int, default=4, 
                            help="number of batch_size")
        parser.add_argument("--epochs", type=int, default=300, 
                            help="number of epochs")
        parser.add_argument("--log_freq", type=int, default=1, 
                            help="per epoch print res")
        parser.add_argument("--seed", default=2022, type=int)

        return parser.parse_args()

    args = get_args()

    D("./datasets/Beauty.txt","./datasets/Beauty_s.txt",10) # 这一行有问题，会重新分割，要删除！

    args.segmented_file = args.data_dir + args.data_name + "_s.txt"

    _,train_seq = get_seqs_and_matrixes("training", args.segmented_file)

    cluster_dataset = DatasetForInDiRec(args, train_seq, data_type="train")
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)

    for batch_idx, data in enumerate(cluster_dataloader):
        print(f"Batch {batch_idx + 1}: {data}")
