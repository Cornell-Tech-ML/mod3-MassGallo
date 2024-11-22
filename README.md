# MiniTorch Module 3

Table of Contents:
- Task 3.1/3.2 Parallel Report
- Task 3.4 Graph
- Task 3.5 HIDDEN = 100
  - Simple
  - Split
  - XOR
- Task 3.5 HIDDEN = 500
  - Simple
  - Split
  - XOR

## Task 3.1/3.2 Parallel Report
```text
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py (163)
--------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                               |
        out: Storage,                                                                       |
        out_shape: Shape,                                                                   |
        out_strides: Strides,                                                               |
        in_storage: Storage,                                                                |
        in_shape: Shape,                                                                    |
        in_strides: Strides,                                                                |
    ) -> None:                                                                              |
        size = len(out)                                                                     |
                                                                                            |
        strides_match = len(out_strides) == len(in_strides) and np.all(                     |
            out_strides == in_strides-------------------------------------------------------| #0
        )                                                                                   |
        shapes_match = len(out_shape) == len(in_shape) and np.all(out_shape == in_shape)----| #1
                                                                                            |
        if strides_match and shapes_match:                                                  |
            for i in prange(size):----------------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                  |
        else:                                                                               |
            for i in prange(size):----------------------------------------------------------| #3
                out_index = np.empty(len(out_shape), dtype=np.int32)                        |
                in_index = np.empty(len(in_shape), dtype=np.int32)                          |
                to_index(i, out_shape, out_index)                                           |
                broadcast_index(out_index, out_shape, in_shape, in_index)                   |
                in_pos = index_to_position(in_index, in_strides)                            |
                out[i] = fn(in_storage[in_pos])                                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(183) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(184) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(216)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py (216)
----------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                     |
        out: Storage,                                                             |
        out_shape: Shape,                                                         |
        out_strides: Strides,                                                     |
        a_storage: Storage,                                                       |
        a_shape: Shape,                                                           |
        a_strides: Strides,                                                       |
        b_storage: Storage,                                                       |
        b_shape: Shape,                                                           |
        b_strides: Strides,                                                       |
    ) -> None:                                                                    |
        size = len(out)                                                           |
                                                                                  |
        strides_match = (                                                         |
            len(out_strides) == len(a_strides)                                    |
            and np.all(out_strides == a_strides)----------------------------------| #4
            and len(out_strides) == len(b_strides)                                |
            and np.all(out_strides == b_strides)----------------------------------| #5
        )                                                                         |
        shapes_match = (                                                          |
            len(out_shape) == len(a_shape)                                        |
            and np.all(out_shape == a_shape)--------------------------------------| #6
            and len(out_shape) == len(b_shape)                                    |
            and np.all(out_shape == b_shape)--------------------------------------| #7
        )                                                                         |
                                                                                  |
        if strides_match and shapes_match:                                        |
            for i in prange(size):------------------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                           |
        else:                                                                     |
            for i in prange(size):------------------------------------------------| #9
                out_index = np.empty(len(out_shape), dtype=np.int32)              |
                a_index = np.empty(len(a_shape), dtype=np.int32)                  |
                b_index = np.empty(len(b_shape), dtype=np.int32)                  |
                                                                                  |
                to_index(i, out_shape, out_index)                                 |
                broadcast_index(out_index, out_shape, a_shape, a_index)           |
                broadcast_index(out_index, out_shape, b_shape, b_index)           |
                a_pos = index_to_position(a_index, a_strides)                     |
                b_pos = index_to_position(b_index, b_strides)                     |
                                                                                  |
                out[int(i)] = fn(a_storage[int(a_pos)], b_storage[int(b_pos)])    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(247) is hoisted out of the parallel loop labelled #9 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(248) is hoisted out of the parallel loop labelled #9 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(249) is hoisted out of the parallel loop labelled #9 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(283)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py (283)
--------------------------------------------------------------------|loop #ID
    def _reduce(                                                    |
        out: Storage,                                               |
        out_shape: Shape,                                           |
        out_strides: Strides,                                       |
        a_storage: Storage,                                         |
        a_shape: Shape,                                             |
        a_strides: Strides,                                         |
        reduce_dim: int,                                            |
    ) -> None:                                                      |
        size = len(out)                                             |
        out_dim = len(out_shape)                                    |
                                                                    |
        for i in prange(size):--------------------------------------| #10
            out_index = np.empty(out_dim, dtype=np.int32)           |
            reduce_size = a_shape[reduce_dim]                       |
            to_index(i, out_shape, out_index)                       |
            a_pos = int(index_to_position(out_index, a_strides))    |
            pos_step = int(a_strides[reduce_dim])                   |
                                                                    |
            acc = out[i]                                            |
            for s in range(reduce_size):                            |
                acc = fn(acc, a_storage[a_pos])                     |
                a_pos += pos_step                                   |
            out[i] = acc                                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(296) is hoisted out of the parallel loop labelled #10 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(out_dim, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py
(311)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/maxg/Documents/CT/MLE/workspace/mod3-MassGallo/minitorch/fast_ops.py (311)
-----------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                           |
    out: Storage,                                                      |
    out_shape: Shape,                                                  |
    out_strides: Strides,                                              |
    a_storage: Storage,                                                |
    a_shape: Shape,                                                    |
    a_strides: Strides,                                                |
    b_storage: Storage,                                                |
    b_shape: Shape,                                                    |
    b_strides: Strides,                                                |
) -> None:                                                             |
    """NUMBA tensor matrix multiply function.                          |
                                                                       |
    Should work for any tensor shapes that broadcast as long as        |
                                                                       |
    ```                                                                |
    assert a_shape[-1] == b_shape[-2]                                  |
    ```                                                                |
                                                                       |
    Optimizations:                                                     |
                                                                       |
    * Outer loop in parallel                                           |
    * No index buffers or function calls                               |
    * Inner loop should have no global writes, 1 multiply.             |
                                                                       |
                                                                       |
    Args:                                                              |
    ----                                                               |
        out (Storage): storage for `out` tensor                        |
        out_shape (Shape): shape for `out` tensor                      |
        out_strides (Strides): strides for `out` tensor                |
        a_storage (Storage): storage for `a` tensor                    |
        a_shape (Shape): shape for `a` tensor                          |
        a_strides (Strides): strides for `a` tensor                    |
        b_storage (Storage): storage for `b` tensor                    |
        b_shape (Shape): shape for `b` tensor                          |
        b_strides (Strides): strides for `b` tensor                    |
                                                                       |
    Returns:                                                           |
    -------                                                            |
        None : Fills in `out`                                          |
                                                                       |
    """                                                                |
    a_batch_stride = int(a_strides[0]) if a_shape[0] > 1 else 0        |
    b_batch_stride = int(b_strides[0]) if b_shape[0] > 1 else 0        |
                                                                       |
    size = len(out)                                                    |
                                                                       |
    for idx in prange(size):-------------------------------------------| #11
        if len(out_shape) > 2:                                         |
            batch = idx // (out_shape[-2] * out_shape[-1])             |
            remainder = idx % (out_shape[-2] * out_shape[-1])          |
        else:                                                          |
            batch = 0                                                  |
            remainder = idx                                            |
                                                                       |
        row = remainder // out_shape[-1]                               |
        col = remainder % out_shape[-1]                                |
                                                                       |
        a_start = int(batch * a_batch_stride + row * a_strides[-2])    |
        b_start = int(batch * b_batch_stride + col * b_strides[-1])    |
                                                                       |
        acc = 0.0                                                      |
        shared_dim = a_shape[-1]                                       |
        a_pos = a_start                                                |
        b_pos = b_start                                                |
                                                                       |
        for k in range(shared_dim):                                    |
            acc += a_storage[a_pos] * b_storage[b_pos]                 |
            a_pos += a_strides[-1]                                     |
            b_pos += b_strides[-2]                                     |
        out[idx] = acc                                                 |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```


## Task 3.4

CPU vs GPU graph using project/timing.py

![Matrix Multiplication Timings for Different Backends](https://github.com/user-attachments/assets/2b147307-054f-43ab-9aa2-a5bc669b0152)


## Task 3.5 HIDDEN = 100
These are CPU backend since it is much faster than the 2-3 seconds GPU.


### Simple

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
```text
Epoch  0 | loss  6.896620141381159 | correct  34 | time per epoch  8.926475048065186
Epoch  10 | loss  1.2352652908907127 | correct  47 | time per epoch  0.27253587245941163
Epoch  20 | loss  0.7240385485856753 | correct  50 | time per epoch  0.27005677223205565
Epoch  30 | loss  1.0458240337840503 | correct  49 | time per epoch  0.2702498912811279
Epoch  40 | loss  0.348366877295593 | correct  50 | time per epoch  0.2707622289657593
Epoch  50 | loss  1.1486769467175253 | correct  50 | time per epoch  0.2703912973403931
Epoch  60 | loss  0.26667076135182655 | correct  50 | time per epoch  0.2915645599365234
Epoch  70 | loss  0.8638985365095069 | correct  50 | time per epoch  0.27204926013946534
Epoch  80 | loss  0.09991902002886789 | correct  50 | time per epoch  0.267558217048645
Epoch  90 | loss  0.7055582938118001 | correct  50 | time per epoch  0.2681692123413086
Epoch  100 | loss  0.01780552379671639 | correct  50 | time per epoch  0.27090590000152587
Epoch  110 | loss  0.6295314044632879 | correct  50 | time per epoch  0.270678448677063
Epoch  120 | loss  0.7547455510084691 | correct  50 | time per epoch  0.272379994392395
Epoch  130 | loss  0.48628625520342506 | correct  50 | time per epoch  0.26944100856781006
Epoch  140 | loss  0.15742074383690147 | correct  50 | time per epoch  0.27213311195373535
Epoch  150 | loss  0.0463289617736913 | correct  50 | time per epoch  0.2714174509048462
Epoch  160 | loss  0.33629826711295646 | correct  50 | time per epoch  0.2699944257736206
Epoch  170 | loss  0.1571489357770421 | correct  50 | time per epoch  0.2770151376724243
Epoch  180 | loss  0.6238053220154176 | correct  50 | time per epoch  0.2761846542358398
Epoch  190 | loss  0.3389570653197701 | correct  50 | time per epoch  0.2712236881256104
Epoch  200 | loss  0.2780251745249778 | correct  50 | time per epoch  0.26734004020690916
Epoch  210 | loss  0.24276105217797056 | correct  50 | time per epoch  0.26871824264526367
Epoch  220 | loss  0.09586949893758596 | correct  50 | time per epoch  0.26805408000946046
Epoch  230 | loss  0.45696076243268974 | correct  50 | time per epoch  0.27120532989501955
Epoch  240 | loss  0.43942752022218284 | correct  50 | time per epoch  0.27094981670379636
Epoch  250 | loss  0.00989315546873658 | correct  50 | time per epoch  0.26809253692626955
Epoch  260 | loss  0.06610723931545025 | correct  50 | time per epoch  0.2682847738265991
Epoch  270 | loss  0.3761612830295597 | correct  50 | time per epoch  0.26919491291046144
Epoch  280 | loss  0.072980320906141 | correct  50 | time per epoch  0.26858010292053225
Epoch  290 | loss  0.0029261190953865247 | correct  50 | time per epoch  0.26903491020202636
Epoch  300 | loss  0.025263558897311553 | correct  50 | time per epoch  0.2697103023529053
Epoch  310 | loss  0.13718565244725475 | correct  50 | time per epoch  0.2691317081451416
Epoch  320 | loss  0.026552805101955075 | correct  50 | time per epoch  0.26780474185943604
Epoch  330 | loss  0.12486464896987834 | correct  50 | time per epoch  0.26760809421539306
Epoch  340 | loss  0.0011390984430940932 | correct  50 | time per epoch  0.27057704925537107
Epoch  350 | loss  0.0008302210454765652 | correct  50 | time per epoch  0.2676544666290283
Epoch  360 | loss  0.00822788475629186 | correct  50 | time per epoch  0.2898097991943359
Epoch  370 | loss  0.16889312934887069 | correct  50 | time per epoch  0.2728488683700562
Epoch  380 | loss  0.2956836222699475 | correct  50 | time per epoch  0.2694361209869385
Epoch  390 | loss  0.27061413325377986 | correct  50 | time per epoch  0.2683246612548828
Epoch  400 | loss  0.044779937863168584 | correct  50 | time per epoch  0.2697275638580322
Epoch  410 | loss  0.05956692569948021 | correct  50 | time per epoch  0.27312602996826174
Epoch  420 | loss  0.0025307810391758468 | correct  50 | time per epoch  0.26969079971313475
Epoch  430 | loss  0.1498813582930123 | correct  50 | time per epoch  0.26987907886505125
Epoch  440 | loss  0.2871739467317299 | correct  50 | time per epoch  0.2694445848464966
Epoch  450 | loss  0.06137953418250558 | correct  50 | time per epoch  0.2717066526412964
Epoch  460 | loss  0.2972794695573593 | correct  50 | time per epoch  0.2717278957366943
Epoch  470 | loss  0.08894245624694344 | correct  50 | time per epoch  0.2724871873855591
Epoch  480 | loss  0.0011009970888083477 | correct  50 | time per epoch  0.27035510540008545
Epoch  490 | loss  0.31282253726431997 | correct  50 | time per epoch  0.2701268196105957
```
### Split

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```
```text
Epoch  0 | loss  5.354309669113101 | correct  32 | time per epoch  9.490080118179321
Epoch  10 | loss  6.164097503446535 | correct  32 | time per epoch  0.2869336843490601
Epoch  20 | loss  5.0022280842027955 | correct  32 | time per epoch  0.28210608959197997
Epoch  30 | loss  4.609370701243754 | correct  44 | time per epoch  0.28189935684204104
Epoch  40 | loss  4.619752527064417 | correct  47 | time per epoch  0.2794911861419678
Epoch  50 | loss  2.9735838312108585 | correct  47 | time per epoch  0.28459887504577636
Epoch  60 | loss  6.0175952929739935 | correct  42 | time per epoch  0.2811363935470581
Epoch  70 | loss  4.089298819942279 | correct  43 | time per epoch  0.3040121078491211
Epoch  80 | loss  4.177641078688299 | correct  47 | time per epoch  0.2857750654220581
Epoch  90 | loss  2.6773516226727128 | correct  48 | time per epoch  0.280540919303894
Epoch  100 | loss  1.5793986461729677 | correct  47 | time per epoch  0.28113858699798583
Epoch  110 | loss  1.9632144581915907 | correct  49 | time per epoch  0.2802300214767456
Epoch  120 | loss  1.66588665712545 | correct  49 | time per epoch  0.27943079471588134
Epoch  130 | loss  1.858687990097871 | correct  49 | time per epoch  0.2829329013824463
Epoch  140 | loss  1.710945879772824 | correct  49 | time per epoch  0.28120348453521726
Epoch  150 | loss  1.7041223749468866 | correct  49 | time per epoch  0.28562791347503663
Epoch  160 | loss  1.400693114337307 | correct  49 | time per epoch  0.2815983533859253
Epoch  170 | loss  0.3180913445514917 | correct  48 | time per epoch  0.2847015857696533
Epoch  180 | loss  0.2447329099195306 | correct  47 | time per epoch  0.2735171318054199
Epoch  190 | loss  1.10517020161022 | correct  49 | time per epoch  0.2681161165237427
Epoch  200 | loss  0.8592591516792861 | correct  50 | time per epoch  0.2893447637557983
Epoch  210 | loss  0.44657743124397675 | correct  50 | time per epoch  0.2710411071777344
Epoch  220 | loss  2.3477557654586687 | correct  48 | time per epoch  0.2693160057067871
Epoch  230 | loss  1.3288444037730744 | correct  50 | time per epoch  0.2670749664306641
Epoch  240 | loss  0.10537100376607161 | correct  47 | time per epoch  0.26847262382507325
Epoch  250 | loss  0.5556283436044728 | correct  50 | time per epoch  0.26648452281951907
Epoch  260 | loss  0.8077398053374479 | correct  50 | time per epoch  0.267962908744812
Epoch  270 | loss  0.7165651341367936 | correct  49 | time per epoch  0.2685586452484131
Epoch  280 | loss  0.9755963421577518 | correct  50 | time per epoch  0.2685638427734375
Epoch  290 | loss  0.30109194939465317 | correct  48 | time per epoch  0.2689292669296265
Epoch  300 | loss  1.9658229793228603 | correct  47 | time per epoch  0.26892457008361814
Epoch  310 | loss  0.5408174264997184 | correct  49 | time per epoch  0.2698298692703247
Epoch  320 | loss  1.0385062134378267 | correct  50 | time per epoch  0.26883468627929685
Epoch  330 | loss  0.27507561959061383 | correct  48 | time per epoch  0.28946120738983155
Epoch  340 | loss  1.1836358006052283 | correct  50 | time per epoch  0.2654029607772827
Epoch  350 | loss  1.0941237602719056 | correct  48 | time per epoch  0.2689235210418701
Epoch  360 | loss  0.2940928744193387 | correct  50 | time per epoch  0.26807665824890137
Epoch  370 | loss  0.2124951422855395 | correct  49 | time per epoch  0.26831305027008057
Epoch  380 | loss  0.23817863041816595 | correct  50 | time per epoch  0.26724820137023925
Epoch  390 | loss  0.10857090732620618 | correct  49 | time per epoch  0.2676910638809204
Epoch  400 | loss  0.27949008696055344 | correct  50 | time per epoch  0.2696254253387451
Epoch  410 | loss  0.25866175029202926 | correct  50 | time per epoch  0.2682978868484497
Epoch  420 | loss  0.5242881527044774 | correct  50 | time per epoch  0.26776134967803955
Epoch  430 | loss  1.958364286821793 | correct  48 | time per epoch  0.2656528472900391
Epoch  440 | loss  0.5345511823775071 | correct  50 | time per epoch  0.270678186416626
Epoch  450 | loss  0.9796319878467457 | correct  49 | time per epoch  0.26813292503356934
Epoch  460 | loss  0.8627176759747535 | correct  48 | time per epoch  0.26789135932922364
Epoch  470 | loss  0.8281149214394261 | correct  50 | time per epoch  0.29334986209869385
Epoch  480 | loss  0.08054717083436663 | correct  50 | time per epoch  0.2688873767852783
Epoch  490 | loss  0.08029101554654741 | correct  50 | time per epoch  0.26792731285095217
```

### XOR

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.02
```
```text
Epoch  0 | loss  8.880384990268741 | correct  21 | time per epoch  8.930634021759033
Epoch  10 | loss  6.124100228451202 | correct  33 | time per epoch  0.2690741539001465
Epoch  20 | loss  3.965378342305107 | correct  46 | time per epoch  0.2730401992797852
Epoch  30 | loss  4.550302594384776 | correct  45 | time per epoch  0.29164605140686034
Epoch  40 | loss  4.398688049803663 | correct  46 | time per epoch  0.26963415145874026
Epoch  50 | loss  4.184008069895223 | correct  46 | time per epoch  0.26714909076690674
Epoch  60 | loss  4.251037265807717 | correct  46 | time per epoch  0.2697781562805176
Epoch  70 | loss  3.1145067202119003 | correct  45 | time per epoch  0.267362117767334
Epoch  80 | loss  5.1032120823918925 | correct  47 | time per epoch  0.2680708408355713
Epoch  90 | loss  3.9344548374470016 | correct  44 | time per epoch  0.26812009811401366
Epoch  100 | loss  2.784846073669136 | correct  47 | time per epoch  0.27305035591125487
Epoch  110 | loss  1.754851796583884 | correct  47 | time per epoch  0.2687010049819946
Epoch  120 | loss  3.8998389254176185 | correct  46 | time per epoch  0.26884260177612307
Epoch  130 | loss  1.4754660834329518 | correct  46 | time per epoch  0.2682874441146851
Epoch  140 | loss  1.2920725753899862 | correct  46 | time per epoch  0.2722918510437012
Epoch  150 | loss  2.5475688326632135 | correct  47 | time per epoch  0.26699302196502683
Epoch  160 | loss  2.247713508305001 | correct  47 | time per epoch  0.29376635551452634
Epoch  170 | loss  1.9651968746047093 | correct  46 | time per epoch  0.2693007230758667
Epoch  180 | loss  2.4448822391594405 | correct  47 | time per epoch  0.26785099506378174
Epoch  190 | loss  1.5698109388374668 | correct  46 | time per epoch  0.26731650829315184
Epoch  200 | loss  2.530202526799047 | correct  47 | time per epoch  0.26565024852752683
Epoch  210 | loss  2.1070828982019667 | correct  47 | time per epoch  0.2695741891860962
Epoch  220 | loss  1.378723351198491 | correct  47 | time per epoch  0.2677305221557617
Epoch  230 | loss  2.3830670559578957 | correct  47 | time per epoch  0.26719667911529543
Epoch  240 | loss  1.3040809993407694 | correct  47 | time per epoch  0.26794729232788084
Epoch  250 | loss  1.7797984123397315 | correct  47 | time per epoch  0.2657289981842041
Epoch  260 | loss  1.1917509104110098 | correct  47 | time per epoch  0.2675101041793823
Epoch  270 | loss  0.898884076966078 | correct  47 | time per epoch  0.27226426601409914
Epoch  280 | loss  0.7592774111756793 | correct  47 | time per epoch  0.26970605850219725
Epoch  290 | loss  0.5372571664875513 | correct  47 | time per epoch  0.290739107131958
Epoch  300 | loss  1.5770493157411223 | correct  48 | time per epoch  0.27109177112579347
Epoch  310 | loss  1.5832163754352688 | correct  47 | time per epoch  0.2673558473587036
Epoch  320 | loss  1.3630237026184497 | correct  47 | time per epoch  0.2695287227630615
Epoch  330 | loss  1.6299651829757391 | correct  48 | time per epoch  0.26834816932678224
Epoch  340 | loss  1.5725071600525846 | correct  48 | time per epoch  0.2689488410949707
Epoch  350 | loss  1.6276365426924013 | correct  48 | time per epoch  0.2693727970123291
Epoch  360 | loss  1.6584879438852178 | correct  48 | time per epoch  0.2691395998001099
Epoch  370 | loss  1.5432971053669609 | correct  49 | time per epoch  0.2691351890563965
Epoch  380 | loss  0.7643061087174929 | correct  48 | time per epoch  0.26947014331817626
Epoch  390 | loss  0.5092505289940763 | correct  49 | time per epoch  0.2670138835906982
Epoch  400 | loss  0.967817655293934 | correct  49 | time per epoch  0.2708153247833252
Epoch  410 | loss  0.608059041858246 | correct  49 | time per epoch  0.26869590282440187
Epoch  420 | loss  1.8567047738640332 | correct  50 | time per epoch  0.28955512046813964
Epoch  430 | loss  1.1941800101869409 | correct  49 | time per epoch  0.2731326580047607
Epoch  440 | loss  1.4046259879179024 | correct  50 | time per epoch  0.2713077783584595
Epoch  450 | loss  0.4059295401822271 | correct  49 | time per epoch  0.2686039924621582
Epoch  460 | loss  0.5132635190405218 | correct  50 | time per epoch  0.2677056550979614
Epoch  470 | loss  0.44545981821648784 | correct  50 | time per epoch  0.2693261861801147
Epoch  480 | loss  1.0161866392770806 | correct  50 | time per epoch  0.2690627336502075
Epoch  490 | loss  0.8852174426171046 | correct  50 | time per epoch  0.26822035312652587
```


## Task 3.5 HIDDEN = 500

### Simple
```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET simple --RATE 0.05
```
```text
Epoch  0 | loss  19.61324853920712 | correct  41 | time per epoch  9.815551996231079
Epoch  10 | loss  0.06657433568794799 | correct  50 | time per epoch  0.767506217956543
Epoch  20 | loss  0.03832086900011532 | correct  50 | time per epoch  0.8365234375
Epoch  30 | loss  0.04440134440717279 | correct  50 | time per epoch  0.9227766752243042
Epoch  40 | loss  0.006306821550474366 | correct  50 | time per epoch  0.8624578714370728
Epoch  50 | loss  0.019570786984732434 | correct  50 | time per epoch  0.7858569383621216
Epoch  60 | loss  0.007020125157924992 | correct  50 | time per epoch  0.8504328966140747
Epoch  70 | loss  0.013354353572077225 | correct  50 | time per epoch  0.9282987833023071
Epoch  80 | loss  0.01432110241889304 | correct  50 | time per epoch  0.9227094173431396
Epoch  90 | loss  0.036247719326003874 | correct  50 | time per epoch  0.8417973041534423
Epoch  100 | loss  0.03409631612030358 | correct  50 | time per epoch  0.8012982845306397
Epoch  110 | loss  0.03484988969433585 | correct  50 | time per epoch  0.8562209129333496
Epoch  120 | loss  0.03786502231160996 | correct  50 | time per epoch  0.9383416414260864
Epoch  130 | loss  0.006890626549032355 | correct  50 | time per epoch  0.9324430465698242
Epoch  140 | loss  0.0068497731744010405 | correct  50 | time per epoch  0.9329792499542237
Epoch  150 | loss  0.01978663346599502 | correct  50 | time per epoch  0.9325888872146606
Epoch  160 | loss  0.009871160156668739 | correct  50 | time per epoch  0.9426707744598388
Epoch  170 | loss  0.018748251580187898 | correct  50 | time per epoch  0.9843989133834838
Epoch  180 | loss  0.028915484890784323 | correct  50 | time per epoch  0.9609874010086059
Epoch  190 | loss  0.033883648340027016 | correct  50 | time per epoch  0.9638368606567382
Epoch  200 | loss  0.01731641694152222 | correct  50 | time per epoch  0.9347399711608887
Epoch  210 | loss  0.025615066165345028 | correct  50 | time per epoch  0.9007247686386108
Epoch  220 | loss  0.028524830292746533 | correct  50 | time per epoch  0.9389621257781983
Epoch  230 | loss  0.004301118655816673 | correct  50 | time per epoch  0.934039044380188
Epoch  240 | loss  0.008567454639264172 | correct  50 | time per epoch  0.9355280637741089
Epoch  250 | loss  0.013400396416942711 | correct  50 | time per epoch  0.9453108310699463
Epoch  260 | loss  0.007424404927888631 | correct  50 | time per epoch  0.9725682973861695
Epoch  270 | loss  7.800867862079468e-05 | correct  50 | time per epoch  0.9362028360366821
Epoch  280 | loss  0.008831497974989883 | correct  50 | time per epoch  0.9271883964538574
Epoch  290 | loss  0.010070112874311865 | correct  50 | time per epoch  0.942905068397522
Epoch  300 | loss  0.007200289097470717 | correct  50 | time per epoch  0.9367838382720948
Epoch  310 | loss  0.03249457687763563 | correct  50 | time per epoch  0.9198525190353394
Epoch  320 | loss  0.010844202002760771 | correct  50 | time per epoch  0.9239262819290162
Epoch  330 | loss  0.008208322162149036 | correct  50 | time per epoch  0.9533308744430542
Epoch  340 | loss  0.009177741182274584 | correct  50 | time per epoch  0.9248940229415894
Epoch  350 | loss  0.020411700369396102 | correct  50 | time per epoch  0.9291349649429321
Epoch  360 | loss  0.007200721004714297 | correct  50 | time per epoch  0.9403388977050782
Epoch  370 | loss  0.00633907496285183 | correct  50 | time per epoch  0.9461596727371215
Epoch  380 | loss  0.007123750780525582 | correct  50 | time per epoch  0.9370569467544556
Epoch  390 | loss  0.005101264088285478 | correct  50 | time per epoch  0.9305283784866333
Epoch  400 | loss  0.008609408606014663 | correct  50 | time per epoch  0.9319384336471558
Epoch  410 | loss  0.019396214080953423 | correct  50 | time per epoch  0.9482777833938598
Epoch  420 | loss  0.0035817643746352944 | correct  50 | time per epoch  0.9469516992568969
Epoch  430 | loss  0.0053524390699428125 | correct  50 | time per epoch  0.9609204292297363
Epoch  440 | loss  0.015211456887993502 | correct  50 | time per epoch  0.9250577449798584
Epoch  450 | loss  0.018346365615458014 | correct  50 | time per epoch  0.9247126340866089
Epoch  460 | loss  0.004629612152922576 | correct  50 | time per epoch  0.8787118196487427
Epoch  470 | loss  0.004905639221436844 | correct  50 | time per epoch  0.9134474039077759
Epoch  480 | loss  0.008500525529747999 | correct  50 | time per epoch  0.9319152593612671
Epoch  490 | loss  0.011076787785084337 | correct  50 | time per epoch  0.9152797937393189
```


## Split

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET split --RATE 0.05
```
```text
Epoch  0 | loss  20.184647645070125 | correct  33 | time per epoch  10.785508155822754
Epoch  10 | loss  4.414931636973566 | correct  44 | time per epoch  0.9472970724105835
Epoch  20 | loss  3.277426757518054 | correct  38 | time per epoch  0.9538928508758545
Epoch  30 | loss  3.8972428215097574 | correct  46 | time per epoch  0.9397274255752563
Epoch  40 | loss  0.8969958548514869 | correct  50 | time per epoch  0.9321231842041016
Epoch  50 | loss  0.3102045295213465 | correct  47 | time per epoch  0.9413138151168823
Epoch  60 | loss  1.9409435922479608 | correct  42 | time per epoch  0.9944133758544922
Epoch  70 | loss  0.2982550365719236 | correct  49 | time per epoch  0.9672640323638916
Epoch  80 | loss  0.2589887221775083 | correct  48 | time per epoch  0.9628942489624024
Epoch  90 | loss  0.4348276552957832 | correct  49 | time per epoch  0.9564194202423095
Epoch  100 | loss  0.5562892341739211 | correct  50 | time per epoch  0.908536696434021
Epoch  110 | loss  0.6289032208619856 | correct  50 | time per epoch  0.9243602991104126
Epoch  120 | loss  0.34900151624868225 | correct  50 | time per epoch  0.9310901165008545
Epoch  130 | loss  1.6902616892847657 | correct  48 | time per epoch  0.9365579605102539
Epoch  140 | loss  0.6474816632824566 | correct  50 | time per epoch  0.9422167301177978
Epoch  150 | loss  0.4586501628344336 | correct  50 | time per epoch  0.9608891010284424
Epoch  160 | loss  0.764723315832154 | correct  50 | time per epoch  0.9426947355270385
Epoch  170 | loss  0.43332695313466196 | correct  50 | time per epoch  0.9187784671783448
Epoch  180 | loss  0.4784002625395225 | correct  50 | time per epoch  0.9321867227554321
Epoch  190 | loss  0.8244842958100701 | correct  50 | time per epoch  0.9534955501556397
Epoch  200 | loss  0.0891295883076137 | correct  50 | time per epoch  0.9380150318145752
Epoch  210 | loss  0.20988131454799436 | correct  50 | time per epoch  0.9372329235076904
Epoch  220 | loss  0.17159614839084564 | correct  50 | time per epoch  0.9476931095123291
Epoch  230 | loss  0.6242818307499518 | correct  50 | time per epoch  0.947040343284607
Epoch  240 | loss  0.14771826567915064 | correct  49 | time per epoch  0.934437370300293
Epoch  250 | loss  0.21688538530596307 | correct  50 | time per epoch  0.9311300992965699
Epoch  260 | loss  0.5134031588687307 | correct  50 | time per epoch  0.9616453409194946
Epoch  270 | loss  0.22820012335996243 | correct  50 | time per epoch  0.9636139154434205
Epoch  280 | loss  0.09472103251662639 | correct  50 | time per epoch  0.9328283548355103
Epoch  290 | loss  0.2705934878615406 | correct  50 | time per epoch  0.9343910694122315
Epoch  300 | loss  0.0995597391947002 | correct  50 | time per epoch  0.943736481666565
Epoch  310 | loss  0.4040431026163583 | correct  50 | time per epoch  0.9428284883499145
Epoch  320 | loss  0.09936909821485478 | correct  50 | time per epoch  0.9480683326721191
Epoch  330 | loss  0.631605651192489 | correct  50 | time per epoch  0.9533915281295776
Epoch  340 | loss  0.5911332394714921 | correct  50 | time per epoch  0.9282742500305176
Epoch  350 | loss  0.04582850316133507 | correct  48 | time per epoch  0.8872334241867066
Epoch  360 | loss  0.13293312029109033 | correct  50 | time per epoch  0.9048324346542358
Epoch  370 | loss  0.07241728784980468 | correct  50 | time per epoch  0.9602488040924072
Epoch  380 | loss  0.07303890675907547 | correct  49 | time per epoch  0.914373779296875
Epoch  390 | loss  0.050054973566620524 | correct  50 | time per epoch  0.9392822504043579
Epoch  400 | loss  0.06470409317308472 | correct  50 | time per epoch  0.7846352577209472
Epoch  410 | loss  0.3794220892394338 | correct  50 | time per epoch  0.7648053169250488
Epoch  420 | loss  0.05987211178693115 | correct  50 | time per epoch  0.7309793949127197
Epoch  430 | loss  0.26504810372108967 | correct  50 | time per epoch  0.6525160074234009
Epoch  440 | loss  0.3453688658351227 | correct  50 | time per epoch  0.6611863374710083
Epoch  450 | loss  0.031746660102198926 | correct  50 | time per epoch  0.6514813899993896
Epoch  460 | loss  0.40794992090189136 | correct  50 | time per epoch  0.6544145107269287
Epoch  470 | loss  0.29299140860066564 | correct  50 | time per epoch  0.6477782011032105
Epoch  480 | loss  0.10125305352924893 | correct  50 | time per epoch  0.6515774250030517
Epoch  490 | loss  0.04466969016502008 | correct  50 | time per epoch  0.6508103847503662
```

## XOR

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET xor --RATE 0.02
```
```text
Epoch  0 | loss  53.100340208488035 | correct  34 | time per epoch  10.858482837677002
Epoch  10 | loss  11.013074297393722 | correct  28 | time per epoch  0.9453609228134155
Epoch  20 | loss  0.7582464125987323 | correct  49 | time per epoch  0.8320117235183716
Epoch  30 | loss  0.8211746085549035 | correct  47 | time per epoch  0.783593726158142
Epoch  40 | loss  0.3037577300682267 | correct  47 | time per epoch  0.8881596326828003
Epoch  50 | loss  0.4365469768380479 | correct  49 | time per epoch  0.9424515008926392
Epoch  60 | loss  0.8583975674136395 | correct  50 | time per epoch  0.9317861795425415
Epoch  70 | loss  0.45791708131739106 | correct  50 | time per epoch  0.8067436695098877
Epoch  80 | loss  1.0843585475860857 | correct  49 | time per epoch  0.8087619543075562
Epoch  90 | loss  0.1868522811840134 | correct  50 | time per epoch  0.8805654048919678
Epoch  100 | loss  0.7474995255394106 | correct  49 | time per epoch  0.9679965257644654
Epoch  110 | loss  1.2902316420452415 | correct  47 | time per epoch  0.9417545795440674
Epoch  120 | loss  0.3474585680243157 | correct  50 | time per epoch  0.9344984292984009
Epoch  130 | loss  0.5251171547636808 | correct  50 | time per epoch  0.9274248838424682
Epoch  140 | loss  0.09658123147931763 | correct  50 | time per epoch  0.9837882280349731
Epoch  150 | loss  0.9159740774960723 | correct  49 | time per epoch  0.9639318466186524
Epoch  160 | loss  0.3714210171285804 | correct  50 | time per epoch  0.9628361940383912
Epoch  170 | loss  1.1884184286871922 | correct  49 | time per epoch  0.9748465299606324
Epoch  180 | loss  0.25965433081668104 | correct  50 | time per epoch  0.9278770923614502
Epoch  190 | loss  0.9445990293634277 | correct  50 | time per epoch  0.9128429889678955
Epoch  200 | loss  0.3528044705605108 | correct  50 | time per epoch  0.9264468669891357
Epoch  210 | loss  0.35927523995244065 | correct  50 | time per epoch  0.9417040348052979
Epoch  220 | loss  0.23872139080824922 | correct  50 | time per epoch  0.9272666215896607
Epoch  230 | loss  0.17629649698838568 | correct  50 | time per epoch  0.9647634983062744
Epoch  240 | loss  0.130531599181125 | correct  50 | time per epoch  0.9811915159225464
Epoch  250 | loss  0.6752779193557062 | correct  50 | time per epoch  0.9360301494598389
Epoch  260 | loss  0.1170651860011246 | correct  50 | time per epoch  0.939432430267334
Epoch  270 | loss  0.4350554398086574 | correct  50 | time per epoch  0.957056975364685
Epoch  280 | loss  0.8225680176724159 | correct  50 | time per epoch  0.9295346021652222
Epoch  290 | loss  0.8426968351813798 | correct  50 | time per epoch  0.9374859094619751
Epoch  300 | loss  0.09180468574923882 | correct  50 | time per epoch  0.9514321327209473
Epoch  310 | loss  0.13102239284945652 | correct  50 | time per epoch  0.9348237037658691
Epoch  320 | loss  0.3229860328975256 | correct  50 | time per epoch  0.943123459815979
Epoch  330 | loss  0.6505419638716206 | correct  50 | time per epoch  0.9579272270202637
Epoch  340 | loss  0.7581776049378873 | correct  50 | time per epoch  0.9518758296966553
Epoch  350 | loss  0.03725778746845675 | correct  50 | time per epoch  0.9479916095733643
Epoch  360 | loss  0.39259374891986787 | correct  50 | time per epoch  0.9370352983474731
Epoch  370 | loss  0.0874962950821108 | correct  50 | time per epoch  0.9260750770568847
Epoch  380 | loss  0.13222930830772647 | correct  50 | time per epoch  0.9550667524337768
Epoch  390 | loss  0.4768494885689849 | correct  50 | time per epoch  0.9537692070007324
Epoch  400 | loss  0.04195588258504414 | correct  50 | time per epoch  0.9616753339767456
Epoch  410 | loss  0.21531263452515542 | correct  50 | time per epoch  0.9610254526138305
Epoch  420 | loss  0.5283238603616838 | correct  50 | time per epoch  0.9318084716796875
Epoch  430 | loss  0.18487596748399732 | correct  50 | time per epoch  0.9057298183441163
Epoch  440 | loss  0.062330287963585584 | correct  50 | time per epoch  0.8772891759872437
Epoch  450 | loss  0.24017402517630915 | correct  50 | time per epoch  0.9697350978851318
Epoch  460 | loss  0.11844257996340139 | correct  50 | time per epoch  0.9204726219177246
Epoch  470 | loss  0.19565678481532328 | correct  50 | time per epoch  0.9266518831253052
Epoch  480 | loss  0.18663171609318752 | correct  50 | time per epoch  0.8320991277694703
Epoch  490 | loss  0.5558801931264334 | correct  50 | time per epoch  0.7622773885726929
```

