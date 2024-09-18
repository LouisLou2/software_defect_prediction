
# 输入的y是DataFrame类型
# name: 数据集的名字
def print_data_distribution_binary(name,y):
    # 取第一列
    assert y.shape[1] == 1
    y_data = y.iloc[:,0]
    print(name)
    total = y.shape[0]
    defective = sum(y_data)
    print("Total: ", total)
    print("Defective: ", defective)
    print("Non-defective: ", total - defective)
    print("Defective Ratio: ", defective/total)
    print()