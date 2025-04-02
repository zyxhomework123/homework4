# homework4

---

### Vocabulary类  
1. **`self.mask_index`**  
   （`mask_token`的索引通过`add_token`方法赋值给`self.mask_index`）  

2. **`self.unk_index`**  
   （对未登录词返回未知词索引）  

3. **`add_token`**  
   （`add_many`循环调用`add_token`逐个添加token）  

---

### CBOWVectorizer类  
4. **`indices`**  
   （当`vector_length < 0`时，向量长度等于实际`indices`的长度）  

5. **`context`** 和 **`target`**  
   （遍历DataFrame中`context`和`target`列的文本内容构建词表）  

6. **`mask_index`**  
   （填充超出`vector_length`的部分为`mask_token`的索引）  

---

### CBOWDataset类  
7. **`str.split()`后的长度**  
   （`_max_seq_length`是所有`context`列分词后的最大长度）  

8. **`indices`** 和 **`target`**  
   （`set_split`选择对应数据集的索引和目标列）  

9. **`target`**  
   （`y_target`通过查找`target`列的token得到）  

---

### 模型结构  
10. **`sum`**  
    （`x_embedded_sum`是嵌入后沿`dim=1`求和）  

11. **`vocab_size`**  
    （输出层`fc1`的`out_features`等于词表大小）  

---

### 训练流程  
12. **`DataLoader`**  
    （通过PyTorch的`DataLoader`类批量加载数据）  

13. **`Dropout`** 和 **`BatchNorm`**  
    （`train()`启用训练模式，激活Dropout和BatchNorm）  

14. **`optimizer`**  
    （反向传播前需清空优化器的梯度）  

15. **`torch.max`**  
    （通过`torch.max`获取预测类别的索引）  

---

### 训练状态管理  
16. **`float('inf')`**  
    （`early_stopping_best_val`初始化为正无穷）  

17. **`args.patience`**  
    （连续`patience`次验证损失未下降触发早停）  

18. **`0`**  
    （验证损失下降时，早停计数器重置为0）  

---

### 设备与随机种子  
19. **`torch.cuda`**  
    （设置CUDA随机种子：`torch.cuda.manual_seed_all(seed)`）  

20. **`torch.cuda`**  
    （`args.device`根据`torch.cuda.is_available()`确定）  

---

### 推理与测试  
21. **`target_word`**  
    （`get_closest`中跳过目标词本身：`if word == target_word: continue`）  

22. **`eval()`**  
    （测试时调用`eval()`禁用Dropout）  

---

### 关键参数  
23. **`0`**  
    （`padding_idx`默认值为0，对应填充符的索引）  

24. **`embedding_dim`**  
    （`args.embedding_dim`控制词向量维度）  

25. **`未减少`**  
    （ReduceLROnPlateau在验证损失**未减少**时触发学习率调整）  

---

### 提交说明  
1. 将答案写入代码仓库的`README.md`，格式按问题顺序排列。  
2. 确保`07 Word2vec.docx`的`word2vec_gensim.ipynb`已按任务要求完成并保留输出结果。  
3. 将两个文档的成果提交至GitHub仓库，文件夹命名为`作业四`。  

--- 

**注意事项**：  
- 调试代码时，优先在PyCharm中设置断点（如`CBOWDataset`的`__getitem__`方法），观察数据加载逻辑。  
- 若遇到CUDA内存不足，可调整`batch_size`参数。  
- 学习率调整策略需结合训练日志分析实际触发条件。
