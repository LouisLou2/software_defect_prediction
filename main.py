# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    tensor = torch.tensor([0.514,0.71,0.1])
    # 大于0.7的元素设置为1，小于0.7的元素设置为0
    tensor = torch.where(tensor>0.7,torch.tensor(1),torch.tensor(0))
    print(tensor)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
