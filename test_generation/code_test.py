def binary_search(arr, low, high, x):
 
    if high >= low:
 
        mid = (high + low) // 2
 
        if arr[mid] == x:
            return mid
 
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
 
        else:
            return binary_search(arr, mid + 1, high, x)
 
    else:
        return -1

def test_binary_search():
    # Test 1: Element in the middle of the array
    arr = [2, 3, 4, 10, 40]
    assert binary_search(arr, 0, len(arr)-1, 10) == 3

    # Test 2: Element at the beginning of the array
    arr = [1, 2, 3, 4, 5]
    assert binary_search(arr, 0, len(arr)-1, 1) == 0

    # Test 3: Element at the end of the array
    arr = [1, 2, 3, 4, 5]
    assert binary_search(arr, 0, len(arr)-1, 5) == 4

    # Test 4: Element not in the array
    arr = [1, 2, 3, 4, 5]
    assert binary_search(arr, 0, len(arr)-1, 6) == -1

    # Test 5: Empty array
    arr = []
    assert binary_search(arr, 0, len(arr)-1, 1) == -1

    # Test 6: Array with one element
    arr = [1]
    assert binary_search(arr, 0, len(arr)-1, 1) == 0

    # Test 7: Array with two elements
    arr = [1, 2]
    assert binary_search(arr, 0, len(arr)-1, 2) == 1

    print("All tests passed")

test_binary_search()