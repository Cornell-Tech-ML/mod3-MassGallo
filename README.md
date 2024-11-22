# MiniTorch Module 3

## Task 3.1/3.2 Parallel Report
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


## Task 3.4

## Task 3.5 CPU

### Simple

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

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

### Split

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

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


### XOR

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.02
```

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


## Task 3.5 CPU

### Simple

```bash
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

### Split

```bash
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```

### XOR

```bash
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.02
```
