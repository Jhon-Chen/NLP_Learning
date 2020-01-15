# LeetCode Notes

## 两数之和

* 哈希表

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap={}
        for ind,num in enumerate(nums):
            hashmap[num] = ind
        for i,num in enumerate(nums):
            j = hashmap.get(target - num)
            if j is not None and i!=j:
                return [i,j]
```

* ![imagef0870461e00dee42.png](https://file.moetu.org/images/2020/01/15/imagef0870461e00dee42.png)

  