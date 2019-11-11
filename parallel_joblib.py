import random
import string

from joblib import Parallel, delayed

random.seed(123)


# define a example function
def rand_string(length):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
        string.ascii_lowercase +
        string.ascii_uppercase +
        string.digits)
                       for i in range(length))
    return rand_str


results = Parallel(n_jobs=4)(delayed(rand_string)(5)
                             for _ in range(10))

print(results)

call = [delayed(rand_string)(5) for _ in range(3)]
print('\n all the calls')
print(call)

single_call = call[0]
print('\n single call')
print(single_call)
print(single_call[0](*single_call[1]))
