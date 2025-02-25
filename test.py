class ListNode:  
    def __init__(self, val=0, next=None):  
        self.val = val  
        self.next = next  
  
def finalListAfterOperations(head):  
    # 用于存储前缀异或和及其第一次出现的位置  
    xor_map = {0: -1}  # 初始前缀异或和为0，没有对应的节点，所以设为None  
    prefix_xor = 0  
    values_to_keep = []  
      
    # 遍历链表，计算前缀异或和  
    current = head  
    while current:  
        prefix_xor ^= current.val  
        print(prefix_xor)
          
        # 如果当前的前缀异或和已经在map中出现过，说明从该位置到当前位置的节点段异或和为0，可以跳过  
        if prefix_xor in xor_map:  
            # 不需要添加任何操作，因为我们已经知道这段可以删除  
            pass  
        else:  
            # 否则，将当前节点的值添加到保留列表中  
            values_to_keep.append(current.val)  
            # 更新前缀异或和第一次出现的位置  
            xor_map[prefix_xor] = current  
          
        current = current.next  
      
    # 根据保留的节点值重新构建链表  
    dummy = ListNode(0)  # 创建一个虚拟头节点  
    current = dummy  
    for val in values_to_keep:  
        current.next = ListNode(val)  
        current = current.next  
      
    # 返回新链表的头节点（跳过虚拟头节点）  

    return dummy.next  
  
# 示例用法  
# 创建一个简单的链表 1 -> 2 -> 3 -> 0  
node1 = ListNode(2)  
node2 = ListNode(3)  
node3 = ListNode(4)  
node4 = ListNode(5)  
node1.next = node2  
node2.next = node3  
node3.next = node4  
  
# 调用函数并打印结果  
result_head = finalListAfterOperations(node1)  
current = result_head  
while current:  
    print(current.val, end=" -> ")  
    current = current.next  
# 注意：这里打印的链表将以None结尾，因为Python的链表打印不会自动停止  
# 你可以通过添加一个条件来避免打印None  
if current is not None:  
    print("None")  
else:  
    print()  # 如果已经是链表末尾，则换行