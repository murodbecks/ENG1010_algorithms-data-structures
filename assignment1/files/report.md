# Drone Collision Prevention

Done by Abror Shopulatov as part of ENG1010 Algorithms and Data Structures course at MBZUAI

## Abstract

Devide and conquer (DnC) method is one of the most important methodologies in Algorithmic Analysis. In this task, we are asked to give a solution to the `top-k closest pair` problem which is primary example of DnC. Problem has 2 variants: finding closest pair and more generalized finding top-k closest pairs. Problem also requires us to test the brute-force and optimized algorithm on variety of settings like varying number of locations or k. In this report, we will highlight data preparations, theoritical basis for baseline solution with O(n^2) complexity and how we optimize our solution to have O(n*logn) complexity. Then, we will show the results from experiments and discuss implications. At the end, we will splend some time on limitations and GenAI usage. So, stay tuned!

## Data preparation

In the task description, we are asked to work with artificially generated 2D and 3D locations. To generate numbers, we used `numpy`'s (\citeharris2020numpy) `numpy.random.uniform` function which give `float` numbers between 0 and n. We set the width of our plane to 10000 as changing makes hardly any difference in our calculations. If we were to choose it too small (compared to number of locations), there is a chance of points coinciding and it may lead to false results. But, 10000 is large enough to avoid such issues.

For given dimension and number of locations, we will generate data and save it as a pandas DataFrame (\citemckinney2010pandas) for further usage.

Problem does not specifies which type of distance we should be using, but hints for real life usage. So, we will stick with Euclidian distance.

And, we will formulate the problem in getting the first occuring k closest pair, so we will favor earlier points in conflicts.

We also implemented helper functions to measure running time and allocated memory of a given function call. We used python's built-in `time` and `tracemalloc` functions (\cite? ). Authors' previous experince come handy here. (https://github.com/murodbecks/yaslis/)

We did most of calculations in numpy functions for faster performance.

## Brute-force solution for top-1

In brute-force solution, we need to look at all the possible combinations in the hope of making sure to get the correct solution. The brute-force solution for this problem seems obvious: looping through every pair and save smallest distance pair. If we observe smaller distance, we update our variables. Here is psudocode:

```brute-force solution for top-1
function find_closest_pair(arr, idx)
    n = len(n)

    initialize closest_pair_info 
    initialize closest_distance

    for i in range(n)
        for j in range(i)
            new_distance = calculate_distance(arr[i], arr[j])

            if new_distance < closest_distance:
                closest_pair_info = new_distance
                closest_distance = new_distance

    return closest_pair_info
```

We are checking every possible combinations, so it is guaranteed to reach a correct solution. 

### Time Complexity analysis

So, for each number until n, we are doing that number of operations. For example, for 4th element, we are calculating distance with 0th, 1st, 2nd, and 3rd elements. So, overall, 

```
0+1+2+...+n = (n+1)*n/2
```

operations. So, it is O(n^2) complexity.

Note that this applies for worst, average and best cases as we are not considering any optimizations.

### Space complexity analysis

For space, we are using array with n elements to store the numbers. And additionally, limited number of space for storing closest_pair_info, closest_distance and new_distance. Since these are constant, we can neglect it. Overall space complexity is O(n)

## Brute-force solution for top-k

Compared to brute-force solution for top-1, we can't throw out smaller distances as we need k of them. Naive solution is to store every distance between pairs and sort out based on distance and get first k elements. In pseudocode, it looks like this:

```brute-force solution for top-k
function find_closest_pair(arr, idx)
    n = len(n)

    initialize all_pair_info 

    for i in range(n)
        for j in range(i)
            new_distance = calculate_distance(arr[i], arr[j])

            closest_pair_info = all_pair_info.append(new_distance)
    
    sorted_all_pair_info = sort_by_distance(all_pair_info)

    return sorted_all_pair_info[:k]
```

### Time Complexity analysis

Similar to top-1 solution, calculating all distances takes O(n^2) time. But, we need to deal with sorting too. Sorting p number of elements takes O(p*logp) complexity. Since we have ~n^2 elements in the list, so the total will be 
```
O(n^2*log(n^2)) = O(2*n^2*logn) = O(n^2*logn)
```

This will end up dominating in overall complexity. 

### Space Complexity analysis

We are storing n*(n-1)/2 elements in a list, so we need O(n^2) space. 

## Slightly optimized Brute-force solution for top-k

We thought about naive solution and decided to optimized a bit more. So, the idea is that we don't need all the elements in the `all_pairs_info` list but only smallest k of them. So, when we observe new distance that is smaller than kth smallest element, we will add it to the list and remove the largest one. In psudocode:


```brute-force solution for top-k
function find_closest_pair(arr)
    n = len(n)

    initialize k_closest_pairs
    initialize kth_closest_distance 

    for i in range(n)
        for j in range(i)
            new_distance = calculate_distance(arr[i], arr[j])

            if len(k_closest_pairs) < k
                k_closest_pairs.append(new_distance)
                initialize kth_closest_distance
            
            elif new_distance < kth_closest_distance
                k_closest_pairs.update(new_distance)
                kth_closest_distance.update_from(k_closest_pairs)
    
    return k_closest_pairs
```

### Time Complexity analysis

Here we need to take care of updating `k_closest_pairs` and `kth_closest_distance` in addition to looping through every pair. Since, `k_closest_pairs` is a list with at most k elements, we can keep it sorted and do insertion in O(k) time. So, in the worst case, we need to update every time and it take O(k)

```
0*k+1*k+2*k+...+n*k = k*(n+1)*n/2
```

Overall complexity will be O(n^2*k). Compared to naive solution, it might not look more efficient. But, we can optimize insertion of the list (with data structures like `heapq`) to decrease insertion complexity to O(logk) then it will be way more efficient considering k<<n.

### Space Complexity analysis

Here our gains shows itself. Since we are not storing giant list, we only need O(k) to store of variables. So, overall O(n+k) which is O(n)


## Optimized solution for top-1 and top-k

The idea came from youtube video (https://youtu.be/6u_hWxbOc7E)

At this point, we were short on time. So, we decided to implement top-k solution and omit top-1 and consider it as a special case of top-k solution.

DnC methaphor starts with considering base-case, that is the case where we obviously know the solution. Since, we can't know the distance of a single location, we need to consider 2 or 3 locations as a base case. We choose 3 as the smallest odd number larger than 1 and there might be the case where we split the data into odd number of elements and 2 can't here.

In the case of 2 points, we can calculate the distance and return it. O(1)
In the case of 3 points, we need to calculate distance between 1st and 2nd, 1st and 3rd, 2nd and 3rd. Overall 3 calculations. this one is also O(1)

Now, there comes division of points. Here we can consider sorting data by X-axis and deviding into to by median. We will call them left and right sides. 

Once we calculate top-k closest values on right and left, we need to combine results. Here we also need to consider combinations for each point to all points in another side. Since, we know `kth_closest_distance` from combination above, we can filter out many comparisons without explicitly doing them. Because, in order to get distance using Euclidian distance, all difference in axis coordinates should be less than `kth_closest_distance`. Otherwise it is guaranteed to get higher value than `kth_closest_distance`. 

We can start filtering from X-axis. As we devided it with median value, all points outside of [median-kth_closest_distance, median+kth_closest_distance] can be considered outside of our interests. Then, comes Y-axis. here we can consider the same thing for every point but we have a danger of repeating ourselves. Because x1 is inside of x2's `kth_closest_distance` territory, x2 is also inside of x1's territory. So, we can sort the filtered array via Y-axis and only look at (y, y+kth_closest_distance] interval. 

For 3D case, we can shrink the space with [z-kth_closest_distance, z+kth_closest_distance]. 

And each region will have limited number of points as we found out that the points on each side is at least `kth_closest_distance` distance away from each other. So, for 2D case, we need to calculate at most 4, and for 3D case we need to calculate at most 8 points. There are independent of number of points, so we can consider them as a linear complexity. 

Here is pseudocode of the algorithm:
```optimized solution for top-k
function closest_k_pairs(arr)
    n = len(arr)

    if n <= 3:
        return base_case(arr)
    
    mid_idx = n//2
    k_pairs_left, kth_closest_distance_left = closest_k_pairs(arr[:mid_idx])
    k_pairs_right, kth_closest_distance_right = closest_k_pairs(arr[mid_idx:])

    k_pairs = combine(k_pairs_left, k_pairs_right)
    kth_closest_distance = min(kth_closest_distance_left, kth_closest_distance_right)

    filtered_arr = points with x coordinates in [arr[mid_idx]-kth_closest_distance, arr[mid_idx]+kth_closest_distance] and sorted by y axis

    for i in range(len(filtered_arr))
        x, y, z = filtered_arr[i]
        egligible_arr = points with y coordinates in (y, y+kth_closest_distance] and z coordiantes in [z-kth_closest_distance, z+kth_closest_distance] if exists

        for arr2 in egligible_arr
            new_distance = distance((x, y, z), arr2)

            if new_distance < kth_closest_distance
                k_pairs.update((x, y, z), arr2)
                kth_closest_distance = new_distance
    
    return k_pairs, kth_closest_distance
```

### Time Complexity analysis

Let's start with sorting array with X-axis. It will take O(n*logn) time.

Calculating base cases, middle points and combining left and right closest pairs takes O(k) time.

In the worst case, filtered_arr covers all points and egligible_arr is consisted of all pairs except pair itself for remaining pairs. If, every pair ended up having less distance, we will spend O(k) time for updating k_closest_pairs. So, it is
`O((n-1)*k+(n-1)*k+...+ 0*k) = O(n^2*k)`
complexity.

Overall:
```
T(n) = 2*T(n/2) + O(n^2*k)
```

So, it will be O(n*logn)

### Space Complexity analysis

Array and sorted array will take O(n) space. 

So, each recursion can take O(n) space and total number of recursions are O(logn). Overall they will be O(n*logn).

Combination process uses constant space so it can be neglected. 

## Dynamic change of locations

Here we can think of smarter solution like tracking which points are changed. ANd we can start with distances from previous run (only taking unchanged ones) and recalculate distances using that number. But it is not guaranteed.

## Experiments

## Limitations

All of the experiments only performed in Apple M4 Max chip. Also, experiments are run only once due to time constraints. Doing on multiple hardware multiple times will give more robust results.

## Conclusion

table summarizing time and space complexities for brute-force, optimized brute force and optimized.

## GenAI usage

```
I hereby declare that the work presented in the submitted report and the accompanying code is entirely my own. No portion of this submission has been copied, reproduced, or directly generated/refined using the responses or outputs of any AI tools (including, but not limited to, ChatGPT, Copilot, Gemini, DeepSeek, or other automated systems) unless stated otherwise. Any external sources, datasets, or tools that have been used are properly cited and referenced. I understand that any breach of this declaration may result in submission cancellation or significant mark deduction.
```

First of all, I respect the policy and tried my best to follow it. But due to lack of time, I had to use LLMs for secondary tasks. For example, I wrote this report in markdown file (`files/report.md`). But due to poor writing skills and lack of experience in latex, I asked LLM to do it for me. I might not do it if there is better support for images in markdown.

Here is the list of chat that I used to complete the task:
- https://t3.chat/share/dwperx2zdu
- https://t3.chat/share/iavuzw5qh3

I take full responsibility and ready for punishment.

