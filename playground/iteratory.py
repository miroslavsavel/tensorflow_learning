# https://www.youtube.com/watch?v=jTYiNjvnHZY

# list - is iterable but it is not iterator
"""
iterable = it can be loop over
    tuple, dicti, strings, files, generators
if something is iterable it has method __iter__()

foor loop in the background is calling __iter__() to get iterator to loop over

what makes something iterator?
iterator is object with state, it remembers where it is during iteration
iterator also know how to get the next value with __next__()
list doesnt have the state and it doesnt have __next__() to get the next value


"""

nums = [1, 2, 3]

i_nums = iter(nums)
# print(i_nums)
# print(dir(i_nums))

# print(next(i_nums))
# print(next(i_nums))
# print(next(i_nums))
# print(next(i_nums))


while True:
    try:
        item = next(i_nums)
        print(item)
    except StopIteration:
        break